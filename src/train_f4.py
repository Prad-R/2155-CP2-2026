"""Does the v4 basis (geodesic + per-cell nonlinear) explain what v1-v3 cannot? LightGBM on z, same folds."""
import argparse, json, os, sys, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.special import ndtri, ndtr
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, folds_for, atomic_save
from features4 import features4
p = argparse.ArgumentParser(); p.add_argument("--procs", type=int, default=8); p.add_argument("--quick", action="store_true"); p.add_argument("--advisors", default="0,1,2,3")
args = p.parse_args()
out = os.path.join(ROOT, "ckpt", "f4" + ("_quick" if args.quick else "")); os.makedirs(out, exist_ok=True)
grids, ratings = load_data(); t0 = time.time()
for a in map(int, args.advisors.split(",")):
    idx = np.where(~np.isnan(ratings[:, a]))[0]
    if args.quick: idx = idx[:500]
    y = ratings[idx, a]; z = ndtri(np.clip(y, 0.0005, 0.9995)); fold = folds_for(len(idx))
    X4, names = features4(grids[idx]); atomic_save(os.path.join(out, f"X_a{a}.npy"), X4)
    f3 = os.path.join(ROOT, "ckpt", "f3", f"X_a{a}.npy")
    sets = {"v4": X4}
    if not args.quick and os.path.exists(f3): sets["v3+v4"] = np.concatenate([np.load(f3), X4], 1)
    print(ADVISORS[a], "v4 features", X4.shape, f"{time.time()-t0:.0f}s", flush=True)
    for nm, X in sets.items():
        oof = np.zeros(len(y), np.float32)
        for k in range(5):
            tr, te = fold != k, fold == k
            m = lgb.LGBMRegressor(n_estimators=150 if args.quick else 6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.25, reg_lambda=2.0, verbose=-1, n_jobs=args.procs, random_state=k)
            m.fit(X[tr], z[tr], eval_set=[(X[te], z[te])], callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[te] = m.predict(X[te])
            if k == 0 and nm == "v4":
                imp = m.booster_.feature_importance("gain"); top = np.argsort(-imp)[:20]
                print("   top gain:", [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in top], flush=True)
        atomic_save(os.path.join(out, f"oof_a{a}_{nm}.npy"), oof); hi = oof >= 0.385
        print(f"   {nm}: R2(z) {r2_score(z, oof):.4f} R2(y) {r2_score(y, ndtr(oof)):.4f} sigma_z_hi {np.sqrt(np.mean((oof[hi]-z[hi])**2)):.4f}  {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
