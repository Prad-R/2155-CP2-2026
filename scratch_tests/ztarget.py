"""Labels look like quantile-transformed raw scores (values k/999). Does modelling the
inverse-normal of the label (a proxy for the raw score) fit better?"""
import os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from common import ROOT, ADVISORS, load_data, folds_for
from features import features, group_cols, INVARIANT
from features2 import features2
grids, ratings = load_data()
n2 = features2(grids[:2])[1]
for a in [int(x) for x in sys.argv[1].split(",")]:
    idx = np.where(~np.isnan(ratings[:, a]))[0]; y = ratings[idx, a]; fold = folds_for(len(idx))
    fx = os.path.join(ROOT, "ckpt", "feat2", f"X_a{a}.npy")
    if os.path.exists(fx): X = np.load(fx)
    else:
        X1, g = features(grids[idx]); X = np.concatenate([X1[:, group_cols(g, INVARIANT)], features2(grids[idx])[0]], 1)
    n1 = X.shape[1] - len(n2)
    pair = np.r_[np.arange(0, 35), n1 + np.array([i for i, s in enumerate(n2) if s.startswith("o")])]
    z = norm.ppf(np.clip(y, 0.0005, 0.9995))
    def cv(fit, cols, target):
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            o[te] = fit(X[tr][:, cols], target[tr], X[te][:, cols], target[te])
        return o
    def lg(Xtr, ttr, Xte, tte):
        m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.015, num_leaves=15, min_child_samples=8, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, n_jobs=4)
        m.fit(Xtr, ttr, eval_set=[(Xte, tte)], callbacks=[lgb.early_stopping(300, verbose=False)]); return m.predict(Xte)
    def rd(Xtr, ttr, Xte, tte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        return RidgeCV(alphas=np.logspace(-2, 3, 12)).fit((Xtr - mu) / sd, ttr).predict((Xte - mu) / sd)
    allc = np.arange(X.shape[1])
    def rep(name, oz=None, oy=None):
        py = norm.cdf(oz) if oz is not None else oy
        hi = py >= 0.65
        print(f"  {name:34s} R2(y) {r2_score(y, py):.4f}  sd near threshold {np.sqrt(np.mean((y[hi]-py[hi])**2)):.4f}" + (f"  R2(z) {r2_score(z, oz):.4f}" if oz is not None else ""), flush=True)
    print("==", ADVISORS[a])
    rep("ridge, counts+pair counts -> z", oz=cv(rd, pair, z))
    rep("ridge, all features -> z", oz=cv(rd, allc, z))
    rep("ridge, all features -> y", oy=cv(rd, allc, y))
    oz = cv(lg, allc, z); rep("LightGBM, all features -> z", oz=oz)
    np.save(os.path.join(ROOT, "ckpt", "feat2", f"oof_a{a}_lgb_z.npy"), oz)
