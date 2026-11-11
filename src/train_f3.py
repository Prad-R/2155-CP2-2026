"""Rule-recovery test: does the wide v3 basis explain an advisor (almost) exactly?
Ridge / Lasso / LightGBM on z, same folds. -> ckpt/f3/X_a{a}.npy, oof_*.npy, results/f3_summary.json"""
import argparse, json, os, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.special import ndtri, ndtr
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, folds_for, atomic_save
from features3 import features3

p = argparse.ArgumentParser(); p.add_argument("--procs", type=int, default=8); p.add_argument("--quick", action="store_true")
args = p.parse_args()
out = os.path.join(ROOT, "ckpt", "f3" + ("_quick" if args.quick else "")); os.makedirs(out, exist_ok=True)
grids, ratings = load_data(); t0 = time.time()
sf = os.path.join(ROOT, "results", "f3_summary" + ("_quick" if args.quick else "") + ".json")
summ = json.load(open(sf)) if os.path.exists(sf) else {}
for a in range(4):
    if ADVISORS[a] in summ:
        continue
    idx = np.where(~np.isnan(ratings[:, a]))[0]
    if args.quick: idx = idx[:600]
    y = ratings[idx, a]; z = ndtri(np.clip(y, 0.0005, 0.9995)); fold = folds_for(len(idx))
    fx = os.path.join(out, f"X_a{a}.npy")
    if os.path.exists(fx): X = np.load(fx); names = json.load(open(os.path.join(out, "names.json")))
    else:
        X, names = features3(grids[idx]); atomic_save(fx, X); json.dump(names, open(os.path.join(out, "names.json"), "w"))
    f2 = os.path.join(ROOT, "ckpt", "feat2", f"X_a{a}.npy")
    X2 = np.load(f2) if (os.path.exists(f2) and not args.quick) else None
    print(ADVISORS[a], "features", X.shape, f"{time.time()-t0:.0f}s", flush=True)
    res, oofs = {}, {}
    def run(name, fit_predict):
        oof = np.zeros(len(y), np.float32)
        for k in range(5):
            tr, te = fold != k, fold == k
            oof[te] = fit_predict(tr, te, k)
        oofs[name] = oof; atomic_save(os.path.join(out, f"oof_a{a}_{name}.npy"), oof)
        hi = oof >= 0.385
        res[name] = dict(r2_z=float(r2_score(z, oof)), r2_y=float(r2_score(y, ndtr(oof))), spearman=float(spearmanr(y, oof)[0]), sigma_z_hi=float(np.sqrt(np.mean((oof[hi] - z[hi]) ** 2))))
        print("  ", name, res[name], f"{time.time()-t0:.0f}s", flush=True)
    def lin(model):
        def f(tr, te, k):
            sc = StandardScaler().fit(X[tr]); m = model().fit(sc.transform(X[tr]), z[tr]); return m.predict(sc.transform(X[te]))
        return f
    run("ridge", lin(lambda: RidgeCV(alphas=np.logspace(-2, 4, 13))))
    run("lasso", lin(lambda: LassoCV(n_alphas=30, cv=3, max_iter=4000, n_jobs=args.procs)))
    def gb(XX):
        def f(tr, te, k):
            m = lgb.LGBMRegressor(n_estimators=200 if args.quick else 6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.3, reg_lambda=2.0, verbose=-1, n_jobs=args.procs, random_state=k)
            m.fit(XX[tr], z[tr], eval_set=[(XX[te], z[te])], callbacks=[lgb.early_stopping(300, verbose=False)])
            if k == 0 and XX is X:
                imp = m.booster_.feature_importance("gain"); top = np.argsort(-imp)[:25]
                print("   top gain:", [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in top], flush=True)
            return m.predict(XX[te])
        return f
    run("lgb_v3", gb(X))
    # trees on the residual of the best linear fit: catches "linear rule + a few nonlinear terms"
    base = oofs["lasso"] if res["lasso"]["r2_z"] > res["ridge"]["r2_z"] else oofs["ridge"]
    zz = z.copy(); z_res = z - base
    def gbres(tr, te, k):
        m = lgb.LGBMRegressor(n_estimators=200 if args.quick else 4000, learning_rate=0.015, num_leaves=7, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, verbose=-1, n_jobs=args.procs, random_state=k)
        m.fit(X[tr], z_res[tr], eval_set=[(X[te], z_res[te])], callbacks=[lgb.early_stopping(300, verbose=False)])
        return base[te] + m.predict(X[te])
    run("lin+lgb_res", gbres)
    if X2 is not None:
        run("lgb_v123", gb(np.concatenate([X2, X], 1)))
    summ[ADVISORS[a]] = res
    json.dump(summ, open(sf + ".tmp", "w"), indent=1); os.replace(sf + ".tmp", sf)
print("done", flush=True)
