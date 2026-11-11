"""Nearest-distance profile model: level ~ counts + sum_{a,b,d} coef[a,b,d] * #(a-cells whose nearest b is at distance d)."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 2
metric = sys.argv[2] if len(sys.argv) > 2 else "man"
D = {"man": MAN, "che": CHE}[metric]
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float); n = len(g)
Of = onehot(g)
dn = near_dist(g, D, self_zero=False, cap=99)            # n,5,49 distance to nearest OTHER cell of type b (99 if none)
C = counts(g)
dmax = 8 if metric == "man" else 6
cols, names = [C[:, 1:]], [f"n{SYM[i]}" for i in range(1, 5)]
for aa in range(5):
    for b in range(5):
        for d in list(range(2, dmax + 1)) + [99]:           # d=1 is the baseline
            if d == dmax:
                v = (Of[:, aa] * ((dn[:, b] >= dmax) & (dn[:, b] < 99))).sum(1)
            else:
                v = (Of[:, aa] * (dn[:, b] == d)).sum(1)
            cols.append(v[:, None]); names.append((aa, b, d))
X = np.hstack(cols + [np.ones((n, 1))])
ci = np.where(cen)[0]; f = folds_for(len(ci), 5, 0)
keep = X[ci].std(0) > 0; keep[-1] = True
X = X[:, keep]; names = [nm for nm, k in zip(names + ["const"], keep) if k]
lam = 1e-3
oof = np.zeros(len(ci))
for k in range(5):
    A = X[ci[f != k]]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t[ci[f != k]])
    oof[f == k] = X[ci[f == k]] @ w
A = X[ci]
w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t[ci])
print(f"{ADVISORS[a]} [{metric}]: nearest-distance profile model {X.shape[1]} cols: OOF resid sd {(t[ci] - oof).std():.3f}  in-sample {(t[ci] - A @ w).std():.3f}")
W = dict(zip(names, w))
print("counts:", {k: round(float(v), 2) for k, v in W.items() if isinstance(k, str)})
print(f"coef by distance d=2..{dmax}+ , none   (relative to d=1)")
for aa in range(5):
    for b in range(5):
        row = [W.get((aa, b, d), np.nan) for d in list(range(2, dmax + 1)) + [99]]
        print(f"  {SYM[aa]} -> nearest {SYM[b]}: " + " ".join(f"{v:+7.2f}" for v in row))
