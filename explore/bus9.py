"""Business: ACE inside the parametric two-regime family to get the raw scale and integer ratios."""
import numpy as np, sys
from lib import *
from blocks import *
from sklearn.isotonic import IsotonicRegression
np.set_printoptions(linewidth=220, suppress=True, precision=3)
g, y, z, lev0, idx = labelled(3)
cen = (y > 0.03) & (y < 0.97)
g, y, lev0 = g[cen], y[cen], lev0[cen]
cnt = counts(g).astype(float); nR, nI, nC, nP, nO = cnt.T
d = 3 * nR - (nI + nC + nO)
I = (g == 1); C = (g == 2)
II = ((I[:, :, :-1] & I[:, :, 1:]).sum((1, 2)) + (I[:, :-1, :] & I[:, 1:, :]).sum((1, 2))).astype(float)
Cleft = C[:, :, 1:].sum((1, 2)).astype(float)       # C cells that have a left neighbour
Cbelow = C[:, :-1, :].sum((1, 2)).astype(float)     # C cells that have a cell below
gap = int(sys.argv[1]) if len(sys.argv) > 1 else 2
m = np.abs(d) >= gap
a = (d >= gap).astype(float)
names = ["A", "A*nI", "A*nC", "A*nP", "A*nO", "B*nI", "B*nC", "B*nP", "B*nO", "II", "C_hasleft", "C_hasbelow"]
X = np.stack([a, a * nI, a * nC, a * nP, a * nO, (1 - a) * nI, (1 - a) * nC, (1 - a) * nP, (1 - a) * nO, II, Cleft, Cbelow, np.ones(len(g))], 1)[m]
u, lev = np.unique(lev0[m], return_inverse=True)
n = len(lev); nl = len(u); wts = np.bincount(lev, minlength=nl).astype(float)
theta = (lev - lev.mean()) / lev.std()
G = X.T @ X + 1e-6 * np.eye(X.shape[1])
for it in range(60):
    w = np.linalg.solve(G, X.T @ theta); p = X @ w
    mm = np.bincount(lev, weights=p, minlength=nl) / wts
    th = IsotonicRegression().fit(np.arange(nl), mm, sample_weight=wts).predict(np.arange(nl))[lev]
    theta = (th - th.mean()) / th.std()
w = np.linalg.solve(G, X.T @ theta); p = X @ w
print(f"|d| >= {gap}: n={n}; ACE resid/sd {np.std(theta - p):.5f}  (R2 {1 - np.var(theta - p):.6f})")
for unit in ("II", "C_hasleft"):
    s = w[names.index(unit)]
    print(f"coefficients in units of {unit}:", {k: round(float(v / s), 3) for k, v in zip(names, w)})
# plain level fit for comparison
t = lev.astype(float); wl = np.linalg.solve(G, X.T @ t)
print("level-linear resid sd", np.std(t - X @ wl).round(3), " ACE resid in level units", (np.std(theta - p) * lev.std()).round(3))
