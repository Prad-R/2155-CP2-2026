"""Transportation: targeted fit of named integer terms, then exactness check and residual hunting."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=3)
g, y, z, lev, idx = labelled(2)
n = len(g)
R, I, C, P, O = [(g == k) for k in range(5)]
cnt = counts(g); has = cnt > 0
left = lambda A, B: (A[:, :, :-1] & B[:, :, 1:]).sum((1, 2))
above = lambda A, B: (A[:, :-1, :] & B[:, 1:, :]).sum((1, 2))
BIG = 9999
dn = near_dist(g, MAN, self_zero=False, cap=BIG)
Of = onehot(g)
d = np.where(dn >= BIG, 0, dn)
F = {
    "dRC": (Of[:, 0] * d[:, 2]).sum(1), "dIR": (Of[:, 1] * d[:, 0]).sum(1), "dIC": (Of[:, 1] * d[:, 2]).sum(1), "dCI": (Of[:, 2] * d[:, 1]).sum(1), "dOR": (Of[:, 4] * d[:, 0]).sum(1),
    "II": left(I, I) + above(I, I), "CC": left(C, C) + above(C, C),
    "nI": cnt[:, 1], "nC": cnt[:, 2], "nP": cnt[:, 3], "nO": cnt[:, 4],
    "R_col0": R[:, :, 0].sum(1), "R_col6": R[:, :, 6].sum(1), "R_row0": R[:, 0, :].sum(1), "R_row6": R[:, 6, :].sum(1),
}
allp = has[:, 0] & has[:, 1] & has[:, 2]
t = lev.astype(float)
ci = np.where((y > 0.08) & (y < 0.92) & allp)[0]
names = list(F)
X = np.stack([F[k] for k in names], 1).astype(float)
o, i, w = cv_fit(X, t, ci, lam=1e-6)
print(f"targeted fit: OOF {o:.4f} in-sample {i:.4f}")
print({k: round(float(v), 3) for k, v in zip(names, w)})
res = t[ci] - (np.hstack([X, np.ones((n, 1))]) @ w)[ci]
print("resid quantiles:", np.quantile(res, [0, .01, .1, .5, .9, .99, 1]).round(2))
o_ = np.argsort(-np.abs(res))[:12]
for j in o_:
    k = ci[j]
    print(f"   res {res[j]:+.2f} label {y[k]:.4f} counts {cnt[k].astype(int)}  {'/'.join(gstr(g[k]))}")
# hunt: regress this residual on blocks
tt = np.zeros(n); tt[ci] = res
B = {"ord4": ordered_pairs(g)[0], "ord8": ordered_pairs(g, ((1, 1), (1, -1)))[0], "pos": Of.reshape(n, -1)[:, 49:],
     "prof_man": near_profile(g, MAN, list(range(1, 9)), "man")[0], "prof_che": near_profile(g, CHE, list(range(1, 7)), "che")[0]}
for combo in (["ord4"], ["pos"], ["prof_man"], ["prof_che"], ["ord4", "prof_man"]):
    Xb = np.hstack([B[k] for k in combo])
    o, i, ww = cv_fit(Xb, tt, ci, lam=1e-2)
    print(f"  resid ~ {'+'.join(combo):30s} OOF {o:6.3f} in-sample {i:6.3f}")
Xb, nb = near_profile(g, MAN, list(range(1, 9)), "man")
o, i, ww = cv_fit(Xb, tt, ci, lam=1e-2)
big = np.argsort(-np.abs(ww[:-1]))[:25]
print("largest residual-profile coefficients:", [(nb[j], round(float(ww[j]), 2)) for j in big])
