"""Is the unexplained part of the labels noise?  Residual distribution shape of low-dimensional level models."""
import numpy as np, sys
from lib import *
from scipy import stats

np.set_printoptions(linewidth=220, suppress=True, precision=2)


def pairs4(g):
    n = len(g); M = np.zeros((n, 5, 5), np.float32)
    for A, B in ((g[:, :, :-1], g[:, :, 1:]), (g[:, :-1, :], g[:, 1:, :])):
        for s in range(5):
            for t in range(5):
                M[:, s, t] += ((A == s) & (B == t)).sum((1, 2))
    M = M + M.transpose(0, 2, 1)
    iu = np.triu_indices(5)
    out = M[:, iu[0], iu[1]].copy(); out[:, iu[0] == iu[1]] /= 2
    return out, [f"{SYM[i]}{SYM[j]}" for i, j in zip(*iu)]


for a in ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]):
    g, y, z, lev, idx = labelled(a)
    cen = (y > 0.08) & (y < 0.92)
    t = lev.astype(float)
    C = counts(g); P, pn = pairs4(g)
    X = np.hstack([C[:, 1:], P[:, :-1], np.ones((len(g), 1))])      # drop one pair col (sum = 84)
    # 5-fold OOF linear fit on central samples
    ci = np.where(cen)[0]; f = folds_for(len(ci), 5, 0)
    oof = np.zeros(len(ci))
    for k in range(5):
        w, *_ = np.linalg.lstsq(X[ci[f != k]], t[ci[f != k]], rcond=None)
        oof[f == k] = X[ci[f == k]] @ w
    res = t[ci] - oof
    w, *_ = np.linalg.lstsq(X[ci], t[ci], rcond=None)
    rin = t[ci] - X[ci] @ w
    print(f"\n===== {ADVISORS[a]}: linear counts+pairs, central n={len(ci)}: OOF resid sd {res.std():.2f}, in-sample {rin.std():.2f}")
    print(f"  skew {stats.skew(res):+.2f}  excess kurtosis {stats.kurtosis(res):+.2f}  (uniform -1.2, normal 0)   range [{res.min():.1f}, {res.max():.1f}]")
    h, e = np.histogram(res, bins=np.arange(-30.5, 31.5, 1))
    print("  hist of residual (bins of 1 level, -30..30):")
    print("   ", h)
    q = np.quantile(res, [.001, .01, .05, .25, .5, .75, .95, .99, .999])
    print("  quantiles:", q.round(1), "  normal-equivalent sd from IQR:", round((q[5] - q[3]) / 1.349, 2))
    # residual vs prediction (heteroscedastic?)
    o = np.argsort(oof)
    for s in np.array_split(o, 6):
        print(f"    pred level ~{oof[s].mean():6.1f}: resid mean {res[s].mean():+5.2f} sd {res[s].std():5.2f}")
