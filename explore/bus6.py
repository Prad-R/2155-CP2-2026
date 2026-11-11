"""Business: is the count function linear inside each labour-market regime (3 nR >= jobs  vs  3 nR < jobs)?"""
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
def fit(m, X, lam=1e-3, show=None):
    A = np.hstack([X[m], np.ones((m.sum(), 1))]); keep = A.std(0) > 0; keep[-1] = True; A = A[:, keep]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t[m])
    r = t[m] - A @ w
    return r.std(), w, keep
for nm, m in (("3nR >= J (workers sufficient)", 3 * nR >= J), ("3nR < J (worker shortage)", 3 * nR < J), ("3nR > J strictly", 3 * nR > J), ("all", np.ones(n, bool))):
    s1, w, keep = fit(m, np.hstack([cnt[:, 1:], loc]))
    s0, w0, _ = fit(m, cnt[:, 1:])
    s2, _, _ = fit(m, np.hstack([cnt[:, 1:], np.stack([cnt[:, i] * cnt[:, j] for i in range(5) for j in range(i, 5)], 1), loc]))
    print(f"{nm:32s} n={m.sum():5d}  sd(level) {t[m].std():6.1f} | counts only {s0:6.2f} | counts + ord4 + prof_man {s1:6.2f} | + cnt2 {s2:6.2f} | count coefs (I,C,P,O rel R): {w0[:4].round(2)}")
