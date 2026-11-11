"""Pairwise interaction map: level ~ counts + sum over (type pair, offset class) coef * pair count."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float)
n = len(g)
O = onehot(g).reshape(n, 5, 7, 7)
cls = {}
for dr in range(0, R + 1):
    for dc in range(-R, R + 1):
        if dr == 0 and dc <= 0:
            continue
        A = O[:, :, :7 - dr, max(0, -dc):7 - max(0, dc)]
        B = O[:, :, dr:, max(0, dc):7 + min(0, dc)]
        M = np.einsum("nahw,nbhw->nab", A, B)
        M = M + M.transpose(0, 2, 1)
        k = (min(dr, abs(dc)), max(dr, abs(dc)))
        cls[k] = cls.get(k, 0) + M
iu = np.triu_indices(5)
C = counts(g)
cols, names = [C[:, 1:]], [f"n{SYM[i]}" for i in range(1, 5)]
keys = sorted(cls, key=lambda k: (k[0] ** 2 + k[1] ** 2, k))
for k in keys:
    M = cls[k]
    for i, j in zip(*iu):
        if (i, j) == (4, 4):
            continue                                    # drop OO as the baseline for every offset (collinear with totals)
        v = M[:, i, j] / (2 if i == j else 1)
        cols.append(v[:, None]); names.append((k, i, j))
X = np.hstack(cols + [np.ones((n, 1))])
ci = np.where(cen)[0]
w, *_ = np.linalg.lstsq(X[ci], t[ci], rcond=None)
res = t[ci] - X[ci] @ w
print(f"{ADVISORS[a]}: pair map up to offset {R}: resid sd {res.std():.2f} ({X.shape[1]} cols)")
print("counts:", dict(zip(names[:4], np.round(w[:4], 2))))
W = {nm: v for nm, v in zip(names[4:], w[4:-1])}
print("coefficients per unordered type pair (rows) and offset class (cols), relative to OO pairs:")
print("pair  " + "  ".join(f"({k[0]},{k[1]})".rjust(7) for k in keys))
for i, j in zip(*iu):
    if (i, j) == (4, 4):
        continue
    print(f" {SYM[i]}{SYM[j]}   " + "  ".join(f"{W[(k, i, j)]:+7.2f}" for k in keys))
