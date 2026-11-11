"""Transportation: test the integer core formula, then search for the missing terms in the residual."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
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
f0 = (-2 * (Of[:, 0] * d[:, 2]).sum(1) - (Of[:, 1] * d[:, 0]).sum(1) - 5 * (Of[:, 1] * d[:, 2]).sum(1) - 5 * (Of[:, 2] * d[:, 1]).sum(1) - (Of[:, 4] * d[:, 0]).sum(1)
      + 3 * (left(I, I) + above(I, I)) + 3 * (left(C, C) + above(C, C)))
allp = has[:, 0] & has[:, 1] & has[:, 2]
print("grids with R, I, C present:", allp.sum())
exact_check(f0[allp], y[allp])
t = lev.astype(float)
ci = np.where((y > 0.08) & (y < 0.92) & allp)[0]
A = np.stack([f0[ci], np.ones(len(ci))], 1)
w, *_ = np.linalg.lstsq(A, t[ci], rcond=None)
res = t[ci] - A @ w
print("level ~ f0: slope", w[0].round(4), "resid sd", res.std().round(3))
tt = t - w[0] * f0
# candidate blocks for the residual
B = {}
B["cnt"] = cnt[:, 1:]
B["ord4"] = ordered_pairs(g)[0]
B["ord8"] = ordered_pairs(g, ((1, 1), (1, -1)))[0]
B["ord2"] = ordered_pairs(g, ((0, 2), (2, 0)))[0]
B["pos"] = Of[:, :, :].reshape(n, -1)[:, 49:]                       # type x position (R baseline)
rowcol = np.concatenate([np.stack([(g == k).sum(2) for k in range(5)], 1).reshape(n, -1), np.stack([(g == k).sum(1) for k in range(5)], 1).reshape(n, -1)], 1)
B["rowcol"] = rowcol                                                  # count of type per row (35) and per column (35)
B["prof_man"] = near_profile(g, MAN, list(range(1, 9)), "man")[0]
B["absent"] = absent(g)[0]
rd = ray_dist(g)                                                       # n,4,5,49
B["raylin"] = np.einsum("nac,ndbc->nadb", Of, rd).reshape(n, -1)       # sum over a-cells of ray distance to b in direction d
B["rayhas"] = np.einsum("nac,ndbc->nadb", Of, (rd > 0).astype(np.float32)).reshape(n, -1)
for combo in (["cnt"], ["cnt", "ord4"], ["cnt", "pos"], ["cnt", "rowcol"], ["cnt", "ord4", "pos"], ["cnt", "ord4", "ord8"], ["cnt", "ord4", "ord2"], ["cnt", "ord4", "prof_man"],
              ["cnt", "ord4", "raylin"], ["cnt", "ord4", "rayhas"], ["cnt", "ord4", "pos", "prof_man"]):
    X = np.hstack([B[k] for k in combo])
    o, i, ww = cv_fit(X, tt, ci, lam=1e-2)
    print(f"  resid ~ {'+'.join(combo):40s} cols {X.shape[1]:4d}  OOF {o:6.3f}  in-sample {i:6.3f}")
X = np.hstack([B["cnt"], B["pos"]])
o, i, ww = cv_fit(X, tt, ci, lam=1e-2)
print("positional maps (I, C, P, O relative to R):")
for k, nm in enumerate("ICPO"):
    print(nm); print(ww[4 + 49 * k:4 + 49 * (k + 1)].reshape(7, 7).round(1))
