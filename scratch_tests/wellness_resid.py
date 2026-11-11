"""What is the Wellness model missing? Residuals of the z0 blend (out-of-fold, z space)."""
import json, os, pickle, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, folds_for
A = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DN = ["Res", "Ind", "Com", "Park", "Off"]; CH = "RICPO"
grids, ratings = load_data()
cfg = json.load(open(os.path.join(ROOT, "results", "blend_z0.json"))); b = cfg["blend"][A]
d = pickle.load(open(os.path.join(ROOT, "ckpt", "gbmz", f"cv_a{A}.pkl"), "rb")); idx, z, y, fold = d["idx"], d["z"], d["y"], d["fold"]
oc = np.load(os.path.join(ROOT, "ckpt", "cnnz", f"a{A}_aug1_oof.npy"))
pred = b["w_gbm"] * d["oof"] + b["w_cnn"] * oc + b["b"]; res = z - pred
g = grids[idx]; X = np.load(os.path.join(ROOT, "ckpt", "feat2", f"X_a{A}.npy"))
print(f"== {ADVISORS[A]}: residual sd {res.std():.3f}; sd by predicted-z band:", {f"{lo}..{hi}": round(float(res[(pred>=lo)&(pred<hi)].std()), 3) for lo, hi in [(-3,-1),(-1,0),(0,.67),(.67,1.3),(1.3,4)]})
# 1. learning curve: is the model data-limited?
for ntr in (1000, 2000, 3000, 4000):
    o = np.zeros(len(z)); rng = np.random.RandomState(0)
    for k in range(5):
        tr = np.where(fold != k)[0]; tr = rng.choice(tr, ntr, replace=False); te = fold == k
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.4, verbose=-1, n_jobs=4).fit(X[tr], z[tr])
        o[te] = m.predict(X[te])
    print(f"  learning curve: {ntr} training grids -> R2(z) {r2_score(z, o):.4f}", flush=True)
# 2. new candidate features (v3) against the residual
n = len(g); f = g.reshape(n, 49); R_, C_ = np.arange(49) // 7, np.arange(49) % 7
DE = np.sqrt((R_[:, None] - R_[None]) ** 2 + (C_[:, None] - C_[None]) ** 2); np.fill_diagonal(DE, 99)
oh = f[:, :, None] == np.arange(5); cnt = oh.sum(1)
F, names = [], []
big = np.zeros((n, 49, 5), bool)                      # cell belongs to a cluster of size >= 3
for i in range(n):
    for t in range(5):
        lab, k = ndimage.label(g[i] == t)
        if k:
            sz = np.bincount(lab.ravel()); big[i, :, t] = (sz[lab] >= 3).ravel() & (lab.ravel() > 0)
for t in range(5):
    de = np.where(oh[:, None, :, t], DE[None], 99).min(2)                 # Euclid distance to nearest t
    db = np.where(big[:, None, :, t], DE[None], 99).min(2); db = np.minimum(db, 9)
    w1 = (oh[:, None, :, t] / (DE[None] ** 2)).sum(2)                      # gravity-style exposure to t
    k2 = (oh[:, None, :, t] & (DE[None] <= 2.01)).sum(2)                   # number of t within radius 2
    for a in range(5):
        ma = oh[:, :, a]; ca = np.maximum(cnt[:, a], 1)
        de_a = np.where(ma, np.minimum(de, 9), np.nan)
        F += [np.nanmean(de_a, 1), np.nanmin(de_a, 1), np.nanstd(de_a, 1), np.nansum(1 / de_a, 1), np.nansum(np.exp(-de_a), 1),
              (db * ma).sum(1) / ca, (w1 * ma).sum(1), (w1 * ma).sum(1) / ca, ((k2 >= 2) * ma).sum(1), ((k2 == 0) * ma).sum(1), (np.minimum(k2, 3) * ma).sum(1)]
        names += [f"{s} [{DN[a]} -> {DN[t]}]" for s in ("mean Euclid dist", "min dist", "sd of dist", "sum 1/dist", "sum exp(-dist)", "mean dist to cluster>=3",
                                                        "sum gravity 1/d^2", "mean gravity", "n with >=2 within r2", "n with none within r2", "sum min(count within r2,3)")]
for t in range(5):
    m = (g == t)
    per = (m[:, :, 1:] != m[:, :, :-1]).sum((1, 2)) + (m[:, 1:] != m[:, :-1]).sum((1, 2))
    F += [per, per / np.maximum(cnt[:, t], 1), (cnt[:, t] ** 2).astype(float)]; names += [f"perimeter {DN[t]}", f"perimeter per cell {DN[t]}", f"n^2 {DN[t]}"]
for a in range(5):
    for t in range(a + 1, 5):
        F += [cnt[:, a] * cnt[:, t], cnt[:, a] / np.maximum(cnt[:, t], 1)]; names += [f"n {DN[a]} x n {DN[t]}", f"n {DN[a]} / n {DN[t]}"]
F = np.nan_to_num(np.stack(F, 1).astype(np.float32), nan=9.0)
rho = np.array([spearmanr(F[:, j], res)[0] if F[:, j].std() > 0 else 0 for j in range(F.shape[1])]); o = np.argsort(-np.abs(rho))[:12]
print("  new features most correlated with the residual:"); [print(f"     {rho[j]:+.3f}  {names[j]}") for j in o]
rho0 = np.array([spearmanr(X[:, j], res)[0] if X[:, j].std() > 0 else 0 for j in range(X.shape[1])]); print(f"  (largest |corr| of residual with a feature already in the model: {np.abs(rho0).max():.3f})")
# 3. can any model predict the residual from old or new features?
for nm, XX in [("features already in the model", X), ("new v3 features", F), ("both", np.concatenate([X, F], 1))]:
    o_ = np.zeros(len(z))
    for k in range(5):
        tr, te = fold != k, fold == k
        o_[te] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=20, subsample=0.7, subsample_freq=1, colsample_bytree=0.3, verbose=-1, n_jobs=4).fit(XX[tr], res[tr]).predict(XX[te])
    print(f"  predicting the residual from {nm}: R2 {r2_score(res, o_):.4f}", flush=True)
# direct refit with v3 added
o_ = np.zeros(len(z)); XX = np.concatenate([X, F], 1)
for k in range(5):
    tr, te = fold != k, fold == k
    m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, n_jobs=4)
    m.fit(XX[tr], z[tr], eval_set=[(XX[te], z[te])], callbacks=[lgb.early_stopping(300, verbose=False)]); o_[te] = m.predict(XX[te])
print(f"  LightGBM refit with v3 added: R2(z) {r2_score(z, o_):.4f} (without: {r2_score(z, d['oof']):.4f})")
np.save(os.path.join(ROOT, "scratch_tests", f"resid_a{A}.npy"), res)
# 4. the worst misses, to look at
for title, order in [("most UNDER-predicted (truth higher)", np.argsort(-res)[:5]), ("most OVER-predicted (truth lower)", np.argsort(res)[:5])]:
    print(f"  {title}:")
    for r0 in range(7):
        print("     " + "   ".join("".join(CH[v] for v in g[i][r0]) for i in order))
    print("     " + "   ".join(f"y{y[i]:.2f}" + f"p{pred[i]:+.1f}"[:5].ljust(2) for i in order))
    print("     counts R/I/C/P/O: " + "  ".join("/".join(str(int(c)) for c in cnt[i]) for i in order))
