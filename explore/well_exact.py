"""Check an explicit integer Wellness formula."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(0)
n = len(g)
R, I, C, P, O = [(g == k) for k in range(5)]
cnt = counts(g)
left = lambda A, B: (A[:, :, :-1] & B[:, :, 1:]).sum((1, 2))        # A left of B
above = lambda A, B: (A[:, :-1, :] & B[:, 1:, :]).sum((1, 2))       # A above B
BIG = 9999
dn = near_dist(g, MAN, self_zero=False, cap=BIG)                    # n,5,49
Of = onehot(g)
has = cnt > 0


def formula(dRC_none=0.0, dRP_none=0.0, dPI_none=0.0):
    dRC = np.where(dn[:, 2] >= BIG, np.nan, dn[:, 2]); dRP = np.where(dn[:, 3] >= BIG, np.nan, dn[:, 3]); dPI = np.where(dn[:, 1] >= BIG, np.nan, dn[:, 1])
    tRC = np.nan_to_num(-3 * dRC ** 2, nan=dRC_none); tRP = np.nan_to_num(-dRP ** 2, nan=dRP_none); tPI = np.nan_to_num(2 * dPI, nan=dPI_none)
    f = (Of[:, 0] * (tRC + tRP)).sum(1) + (Of[:, 3] * tPI).sum(1)
    f = f + 20 * left(P, R) + 20 * left(P, P) + 15 * above(R, P) + 20 * above(P, P) + 4 * above(R, I)
    return f


f = formula()
allp = has[:, 1] & has[:, 2] & has[:, 3]
print("grids with I, C and P all present:", allp.sum())
exact_check(f[allp], y[allp])
vals, inv = np.unique(np.round(y[allp], 7), return_inverse=True)
fa = f[allp]
lo = np.full(len(vals), np.inf); hi = np.full(len(vals), -np.inf)
np.minimum.at(lo, inv, fa); np.maximum.at(hi, inv, fa)
bad = np.where(lo != hi)[0]
print("levels with spread:", len(bad))
for k in bad[:15]:
    u, c = np.unique(fa[inv == k], return_counts=True)
    print(f"   label {vals[k]:.5f}: f values {dict(zip(u.astype(int), c))}")
cen = (vals > 0.08) & (vals < 0.92)
d = np.diff(lo[cen])
print("central steps of f between consecutive levels:", np.unique(d, return_counts=True))
# fit refinement: regress residual of level on everything to find missing pieces
ci = np.where((y > 0.08) & (y < 0.92) & allp)[0]
t = lev.astype(float)
A = np.stack([f[ci], np.ones(len(ci))], 1)
w, *_ = np.linalg.lstsq(A, t[ci], rcond=None)
res = t[ci] - A @ w
print("level ~ f: slope", w[0].round(4), "resid sd", res.std().round(3), " max |res|", np.abs(res).max().round(2))
o = np.argsort(-np.abs(res))[:10]
for j in o:
    print(f"   res {res[j]:+.2f} label {y[ci[j]]:.4f} f {f[ci[j]]:.0f} counts {cnt[ci[j]].astype(int)}  {'/'.join(gstr(g[ci[j]]))}")
# grids lacking a type
for nm, col in (("I", 1), ("C", 2), ("P", 3)):
    m = ~has[:, col]
    print(f"grids without {nm}: {m.sum()}  labels: {np.sort(y[m]).round(4)[:30]}")
