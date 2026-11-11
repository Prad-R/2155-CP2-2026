"""Business: pooled fit with regime-specific count coefficients and shared local terms; locate the regime boundary."""
import numpy as np, sys
from lib import *
from blocks import *
np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(3)
cen = (y > 0.05) & (y < 0.95)
g, y, lev = g[cen], y[cen], lev[cen]
n = len(g); t = lev.astype(float)
cnt = counts(g).astype(float); nR, nI, nC, nP, nO = cnt.T
J = nI + nC + nO
Xo, no = ordered_pairs(g)
Xp, npn = near_profile(g, MAN, list(range(1, 9)), "man")
loc = np.hstack([Xo, Xp])
def fit(X, lam=1e-3):
    A = np.hstack([X, np.ones((n, 1))]); keep = A.std(0) > 0; keep[-1] = True; A = A[:, keep]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t)
    return (t - A @ w), w, keep
for off in (-3, -2, -1, 0, 1, 2, 3):
    a = (3 * nR - J >= off).astype(float)[:, None]
    X = np.hstack([a, a * cnt[:, 1:], (1 - a) * cnt[:, 1:], loc])
    r, w, keep = fit(X)
    Xl = np.hstack([a, a * cnt[:, 1:], (1 - a) * cnt[:, 1:], a * loc, (1 - a) * loc])
    r2, _, _ = fit(Xl)
    print(f"regime A := 3nR - J >= {off:+d}: n_A {int(a.sum()):5d}  shared local terms: resid {r.std():6.2f} | regime-specific local terms: resid {r2.std():6.2f}")
d = 3 * nR - J
a = (d >= 0).astype(float)[:, None]
X = np.hstack([a, a * cnt[:, 1:], (1 - a) * cnt[:, 1:], loc])
r, w, keep = fit(X)
print("residual by d = 3nR - J (shared-local, boundary at 0):")
for v in range(-12, 13):
    m = d == v
    if m.sum() >= 5:
        print(f"   d={v:+3d} n={m.sum():4d} mean resid {r[m].mean():+6.2f} sd {r[m].std():5.2f}")
