"""Experiment: do richer invariant features / non-CNN models predict the advisors better?

5-fold OOF (same folds as gbm0/cnn0) for LightGBM and an MLP on features v1+v2.
Writes ckpt/<tag>/oof_a{a}_{model}.npy and results/<tag>_summary.json. Resumable per unit.
"""
import argparse, json, os, pickle
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
import torch, torch.nn as nn
from sklearn.metrics import r2_score

from common import ROOT, ADVISORS, load_data, folds_for, atomic_save
from features import features, group_cols, INVARIANT
from features2 import features2

p = argparse.ArgumentParser()
p.add_argument("--tag", default="feat2"); p.add_argument("--procs", type=int, default=4)
p.add_argument("--advisors", default="0,1,2,3")
args = p.parse_args()
out = os.path.join(ROOT, "ckpt", args.tag); os.makedirs(out, exist_ok=True)
grids, ratings = load_data()
cfg = json.load(open(os.path.join(ROOT, "results", "blend_b0.json")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = lambda *a: print(*a, flush=True)


def lgbm(Xtr, ytr, Xte, yte, seed):
    m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, n_jobs=args.procs, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xte)


def mlp(Xtr, ytr, Xte, yte, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, Bt = [torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev) for x in (Xtr, Xte)]
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    pred = 0
    for s in range(3):
        torch.manual_seed(seed * 10 + s)
        net = nn.Sequential(nn.Dropout(0.1), nn.Linear(A.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.GELU(),
                            nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
        ep = 300; opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=ep * (len(A) // 128 + 1))
        for e in range(ep):
            net.train(); perm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 128):
                b = perm[i:i + 128]
                loss = nn.functional.smooth_l1_loss(net(A[b]).squeeze(1), yt[b], beta=0.05)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): pred = pred + net(Bt).squeeze(1).cpu().numpy() / 3
    return pred


def stats(y, pr):
    hi = pr >= 0.65
    return dict(r2=round(float(r2_score(y, pr)), 4), sigma_hi=round(float(np.sqrt(np.mean((y[hi] - pr[hi]) ** 2))), 4))


summary = {}
for a in [int(x) for x in args.advisors.split(",")]:
    idx = np.where(~np.isnan(ratings[:, a]))[0]; y = ratings[idx, a]; fold = folds_for(len(idx))
    fx = os.path.join(out, f"X_a{a}.npy")
    if os.path.exists(fx): X = np.load(fx)
    else:
        X1, groups = features(grids[idx]); X2, n2 = features2(grids[idx])
        X = np.concatenate([X1[:, group_cols(groups, INVARIANT)], X2], 1); atomic_save(fx, X)
    n1 = X.shape[1] - len(features2(grids[:2])[1])
    oofs = {}
    d = pickle.load(open(os.path.join(ROOT, "ckpt", cfg["gbm"], f"cv_a{a}_all.pkl"), "rb")); assert np.allclose(d["y"], y)
    oofs["old_gbm"] = d["oof"]; oofs["old_cnn"] = np.load(os.path.join(ROOT, "ckpt", cfg["cnn"], f"a{a}_aug{cfg['augs'][a]}_oof.npy"))
    for name, fn, cols in [("lgb_inv1", lgbm, np.arange(n1)), ("lgb_inv12", lgbm, np.arange(X.shape[1])), ("lgb_v2only", lgbm, np.arange(n1, X.shape[1])),
                           ("mlp_inv12", mlp, np.arange(X.shape[1]))]:
        f = os.path.join(out, f"oof_a{a}_{name}.npy")
        if os.path.exists(f): oofs[name] = np.load(f); continue
        o = np.zeros(len(y), np.float32)
        for k in range(5):
            tr, te = fold != k, fold == k
            o[te] = fn(X[tr][:, cols], y[tr], X[te][:, cols], y[te], k)
        atomic_save(f, o); oofs[name] = o
        P(ADVISORS[a], name, stats(y, o))
    res = {k: stats(y, v) for k, v in oofs.items()}
    for bn, ks in [("blend_old", ["old_gbm", "old_cnn"]), ("blend_new", ["lgb_inv12", "mlp_inv12", "old_cnn"]), ("blend_all", list(oofs))]:
        A = np.stack([oofs[k] for k in ks] + [np.ones(len(y))], 1); co = np.linalg.lstsq(A, y, rcond=None)[0]
        res[bn] = stats(y, A @ co); res[bn]["w"] = {k: round(float(c), 3) for k, c in zip(ks, co)}
    summary[ADVISORS[a]] = res
    P("==", ADVISORS[a]); [P("   ", k, v) for k, v in res.items()]
    json.dump(summary, open(os.path.join(ROOT, "results", f"{args.tag}_summary.json"), "w"), indent=1)
