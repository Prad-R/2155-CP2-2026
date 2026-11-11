"""Exploration + LightGBM surrogates (5-fold CV per advisor) + prediction on all grids.

Resumable: every unit (feature cache, per-advisor/feature-set CV, full prediction)
is written atomically and skipped if present.

  python -u src/train_gbm.py --tag gbm0 [--quick]
"""
import argparse, json, os, pickle, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import r2_score

from common import ROOT, ADVISORS, THRESH, load_data, atomic_save, folds_for
from features import features, features_parallel, group_cols, INVARIANT

p = argparse.ArgumentParser()
p.add_argument("--tag", default="gbm0")
p.add_argument("--quick", action="store_true", help="tiny end-to-end test")
p.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
args = p.parse_args()

out = os.path.join(ROOT, "ckpt", args.tag + ("_quick" if args.quick else ""))
res = os.path.join(ROOT, "results")
os.makedirs(out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

grids, ratings = load_data()
N = len(grids)
lab = ~np.isnan(ratings)
log("grids", grids.shape, "labels per advisor", lab.sum(0))

# ---------------------------------------------------------------- exploration
ex = {}
ex["n_labelled_by_k_advisors"] = {int(k): int(v) for k, v in zip(*np.unique(lab.sum(1), return_counts=True))}
all4 = lab.all(1)
ex["n_all4_labelled"] = int(all4.sum())
ex["n_all4_valid"] = int((ratings[all4] >= THRESH).all(1).sum())
ex["frac_valid_per_advisor"] = [float((ratings[lab[:, a], a] >= THRESH).mean()) for a in range(4)]
corr, npair, both = {}, {}, {}
for a in range(4):
    for b in range(a + 1, 4):
        m = lab[:, a] & lab[:, b]
        k = f"{ADVISORS[a]}-{ADVISORS[b]}"
        npair[k] = int(m.sum())
        if m.sum() > 10:
            corr[k] = float(np.corrcoef(ratings[m, a], ratings[m, b])[0, 1])
            both[k] = float(((ratings[m, a] >= THRESH) & (ratings[m, b] >= THRESH)).mean())
ex["pair_overlap"], ex["pair_corr"], ex["pair_both_valid_frac"] = npair, corr, both
cnt = (grids.reshape(N, 49)[:, :, None] == np.arange(5)).sum(1)
ex["mean_counts_all"] = cnt.mean(0).tolist()
ex["mean_counts_valid"] = [cnt[lab[:, a] & (ratings[:, a] >= THRESH)].mean(0).tolist() for a in range(4)]
ex["score_unique_values"] = [int(len(np.unique(ratings[lab[:, a], a]))) for a in range(4)]
ex["score_quantiles_0_25_50_75_100"] = [np.quantile(ratings[lab[:, a], a], [0, .25, .5, .75, 1]).tolist() for a in range(4)]
ex["n_unique_grids_in_100k_sample"] = int(len(np.unique(grids[:100000].reshape(-1, 49), axis=0)))
log("explore", json.dumps(ex, indent=1))

# ---------------------------------------------------------------- features
_, groups = features(grids[:4])
fpath = os.path.join(ROOT, "cache", "feats_quick.npy" if args.quick else "feats_all.npy")
sel = np.arange(N)
if args.quick:
    sel = np.unique(np.concatenate([np.where(lab.any(1))[0][:3000], np.arange(2000)]))
if os.path.exists(fpath):
    X = np.load(fpath, mmap_mode="r")
else:
    log("computing features for", len(sel), "grids with", args.procs, "procs")
    X = features_parallel(grids[sel], procs=args.procs)
    atomic_save(fpath, X)
log("features", X.shape, {k: v[1] - v[0] for k, v in groups.items()})
pos = -np.ones(N, int)
pos[sel] = np.arange(len(sel))

FSETS = {"all": list(groups), "inv": INVARIANT}
params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
              n_estimators=300 if args.quick else 4000, verbose=-1, n_jobs=args.procs)

# ---------------------------------------------------------------- CV
summary = {}
for a in range(4):
    idx = np.where(lab[:, a] & (pos >= 0))[0]
    y = ratings[idx, a]
    fold = folds_for(len(idx))
    for fs, names in FSETS.items():
        unit = os.path.join(out, f"cv_a{a}_{fs}.pkl")
        if os.path.exists(unit):
            d = pickle.load(open(unit, "rb"))
        else:
            cols = group_cols(groups, names)
            Xa = np.asarray(X[pos[idx]][:, cols])
            oof = np.zeros(len(idx), np.float32)
            models = []
            for k in range(5):
                tr, te = fold != k, fold == k
                m = lgb.LGBMRegressor(**params, random_state=k)
                m.fit(Xa[tr], y[tr], eval_set=[(Xa[te], y[te])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[te] = m.predict(Xa[te])
                models.append(m)
            imp = np.mean([m.booster_.feature_importance("gain") for m in models], 0)
            d = dict(idx=idx, y=y, oof=oof, fold=fold, models=models, cols=cols, imp=imp)
            tmp = unit + ".tmp"
            pickle.dump(d, open(tmp, "wb"))
            os.replace(tmp, unit)
        oof, r2 = d["oof"], r2_score(y, d["oof"])
        stats = {"r2": float(r2), "rmse": float(np.sqrt(np.mean((oof - y) ** 2)))}
        for mrg in (0.75, 0.80, 0.85, 0.90):
            pv = oof >= mrg
            stats[f"n_pred>={mrg}"] = int(pv.sum())
            stats[f"prec@{mrg}"] = float((y[pv] >= THRESH).mean()) if pv.any() else None
        hi = y >= 0.6
        stats["rmse_y>=0.6"] = float(np.sqrt(np.mean((oof[hi] - y[hi]) ** 2)))
        # top features by gain
        names_flat = []
        for g_ in names:
            s, e = groups[g_]
            names_flat += [f"{g_}{j}" for j in range(e - s)]
        top = np.argsort(-d["imp"])[:12]
        stats["top_feats"] = [f"{names_flat[j]}:{d['imp'][j]/d['imp'].sum():.3f}" for j in top]
        summary[f"{ADVISORS[a]}/{fs}"] = stats
        log(ADVISORS[a], fs, json.dumps(stats))

json.dump({"explore": ex, "gbm_cv": summary, "groups": groups},
          open(os.path.join(res, f"{args.tag}{'_quick' if args.quick else ''}_summary.json"), "w"), indent=1)

# ---------------------------------------------------------------- predict all
ppath = os.path.join(out, "pred_all.npy")  # (n_sel, 4 advisors, 5 folds), feature set "all"
if not os.path.exists(ppath):
    P = np.zeros((len(sel), 4, 5), np.float32)
    for a in range(4):
        d = pickle.load(open(os.path.join(out, f"cv_a{a}_all.pkl"), "rb"))
        for s in range(0, len(sel), 50000):
            Xc = np.asarray(X[s:s + 50000][:, d["cols"]])
            for k, m in enumerate(d["models"]):
                P[s:s + 50000, a, k] = m.predict(Xc)
        log("predicted advisor", a)
    atomic_save(ppath, P)
P = np.load(ppath)
mn = P.mean(2).min(1)
log("dataset grids with predicted min-advisor >= ", {m: int((mn >= m).sum()) for m in (0.75, 0.8, 0.85, 0.9)})
open(os.path.join(out, "DONE"), "w").write("ok\n")
log("DONE")
