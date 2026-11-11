"""Check an explicit integer Tax formula: is label a monotone function of it with zero scatter?"""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(1)
n = len(g)
R, I, C, P, O = [(g == k) for k in range(5)]
cnt = counts(g)
left = lambda A, B: (A[:, :, :-1] & B[:, :, 1:]).sum((1, 2))        # A left of B
above = lambda A, B: (A[:, :-1, :] & B[:, 1:, :]).sum((1, 2))       # A above B
feats = {
    "nR": cnt[:, 0], "nI": cnt[:, 1], "nC": cnt[:, 2], "nP": cnt[:, 3], "nO": cnt[:, 4],
    "RI": left(R, I) + left(I, R) + above(R, I) + above(I, R),
    "P_west_of_R": left(P, R), "P_east_of_R": left(R, P), "P_south_of_R": above(R, P), "P_north_of_R": above(P, R),
}
coef = {"nR": 10, "nI": 0, "nC": 3, "nP": -20, "nO": 0, "RI": -2, "P_west_of_R": 8, "P_east_of_R": 3, "P_south_of_R": 10, "P_north_of_R": 3}
f = sum(coef[k] * feats[k] for k in coef)
vals, inv = np.unique(y, return_inverse=True)
print("spearman(label, f) =", np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(f)))[0, 1])
# per level: distinct f values
spread = np.array([f[inv == k].max() - f[inv == k].min() for k in range(len(vals))])
mean = np.array([f[inv == k].mean() for k in range(len(vals))])
print("levels:", len(vals), " levels with a single f value:", (spread == 0).sum(), " max spread:", spread.max())
cen = (vals > 0.08) & (vals < 0.92)
d = np.diff(mean[cen])
print("central: mean step of f between consecutive levels:", d.mean().round(3), " step histogram:", np.unique(np.round(d, 1), return_counts=True))
# OLS refit of level on these features (central) to see the scale and residual
ci = (y > 0.08) & (y < 0.92)
X = np.stack([feats[k] for k in coef if k != "nR"] + [np.ones(n)], 1).astype(float)
w, *_ = np.linalg.lstsq(X[ci], lev[ci].astype(float), rcond=None)
res = lev[ci] - X[ci] @ w
print("OLS level ~ feats:", dict(zip([k for k in coef if k != "nR"], w.round(3))), " resid sd", res.std().round(3))
# residual of level vs f: where are the misfits?
for lo, hi in ((0, .02), (.02, .08), (.08, .3), (.3, .7), (.7, .92), (.92, .98), (.98, 1.01)):
    m = (vals >= lo) & (vals < hi)
    print(f"  label [{lo},{hi}): levels {m.sum():3d}  single-valued {int((spread[m] == 0).sum()):3d}  mean spread {spread[m].mean():.2f}")
# show a few central levels with spread
bad = np.where(cen & (spread > 0))[0][:12]
for k in bad:
    u, c = np.unique(f[inv == k], return_counts=True)
    print(f"   level {k} label {vals[k]:.4f}: f values {dict(zip(u.astype(int), c))}")
