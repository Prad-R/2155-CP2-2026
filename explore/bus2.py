"""Business: shape of the count dependence. Additive 1-D functions, then pairwise tables."""
import numpy as np, sys
from lib import *
from blocks import *
import lightgbm as lgb

np.set_printoptions(linewidth=220, suppress=True, precision=1)
g, y, z, lev, idx = labelled(3)
n = len(g)
cnt = counts(g).astype(int)
t = z * 100
f = folds_for(n, 5, 0)


def gbm_oof(X, t, **kw):
    oof = np.zeros(n)
    for k in range(5):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, **kw)
        m.fit(X[f != k], t[f != k]); oof[f == k] = m.predict(X[f == k])
    return oof


o1 = gbm_oof(cnt, t)
print(f"GBM on 5 counts: OOF resid sd {np.std(t - o1):.2f} of {t.std():.2f} (R2 {1 - np.var(t - o1) / t.var():.4f})")
Xo = ordered_pairs(g)[0]
o2 = gbm_oof(np.hstack([cnt, Xo]), t)
print(f"GBM on counts + ordered N4 pairs: OOF resid sd {np.std(t - o2):.2f} (R2 {1 - np.var(t - o2) / t.var():.4f})")
# additive shapes
for k in range(5):
    v = cnt[:, k]
    print(f"mean z*100 by n{SYM[k]}:", " ".join(f"{c}:{t[v == c].mean():+.0f}" for c in range(0, 26) if (v == c).sum() >= 8))
# 2-D table of mean z for the two strongest: nC vs nI, nR vs nC  (bins)
bins = [0, 3, 5, 7, 9, 11, 13, 15, 18, 22, 50]
for (i, j) in ((0, 2), (0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (0, 4), (2, 4)):
    print(f"mean z*100: rows n{SYM[i]} bins, cols n{SYM[j]} bins {bins}")
    bi = np.digitize(cnt[:, i], bins) - 1; bj = np.digitize(cnt[:, j], bins) - 1
    for a in range(len(bins) - 1):
        row = []
        for b in range(len(bins) - 1):
            m = (bi == a) & (bj == b)
            row.append(f"{t[m].mean():+5.0f}" if m.sum() >= 5 else "    .")
        print(f"   n{SYM[i]} in [{bins[a]},{bins[a + 1]}): " + " ".join(row))
