"""LightGBM on invariant features v1+v2, target = inverse-normal of the label (z).
5-fold, same folds as everything else. ckpt/<tag>/cv_a{a}.pkl = dict(y, z, oof (z), models)."""
import argparse, os, pickle
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.special import ndtri, ndtr
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, folds_for
from features2 import features_inv2

p = argparse.ArgumentParser()
p.add_argument("--tag", default="gbmz"); p.add_argument("--procs", type=int, default=4)
args = p.parse_args()
out = os.path.join(ROOT, "ckpt", args.tag); os.makedirs(out, exist_ok=True)
grids, ratings = load_data()
for a in range(4):
    unit = os.path.join(out, f"cv_a{a}.pkl")
    if os.path.exists(unit):
        continue
    idx = np.where(~np.isnan(ratings[:, a]))[0]; y = ratings[idx, a]; z = ndtri(np.clip(y, 0.0005, 0.9995)); fold = folds_for(len(idx))
    fx = os.path.join(ROOT, "ckpt", "feat2", f"X_a{a}.npy")
    X = np.load(fx) if os.path.exists(fx) else features_inv2(grids[idx])
    oof, models = np.zeros(len(y), np.float32), []
    for k in range(5):
        tr, te = fold != k, fold == k
        m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, n_jobs=args.procs, random_state=k)
        m.fit(X[tr], z[tr], eval_set=[(X[te], z[te])], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[te] = m.predict(X[te]); models.append(m)
    pickle.dump(dict(idx=idx, y=y, z=z, oof=oof, fold=fold, models=models), open(unit + ".tmp", "wb")); os.replace(unit + ".tmp", unit)
    hi = oof >= 0.385
    print(ADVISORS[a], f"R2(z) {r2_score(z, oof):.4f} R2(y) {r2_score(y, ndtr(oof)):.4f} sigma_z(hi) {np.sqrt(np.mean((oof[hi]-z[hi])**2)):.4f}", flush=True)
