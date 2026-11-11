"""Business: coefficient tables inside each regime (d = 3nR - J <= -2  and  d >= +2)."""
import numpy as np, sys
from lib import *
from blocks import *
np.set_printoptions(linewidth=220, suppress=True, precision=2)
g, y, z, lev, idx = labelled(3)
cen = (y > 0.05) & (y < 0.95)
g, y, lev = g[cen], y[cen], lev[cen]
n = len(g); t = lev.astype(float)
cnt = counts(g).astype(float); nR, nI, nC, nP, nO = cnt.T
d = 3 * nR - (nI + nC + nO)
Xo, no = ordered_pairs(g)
dv = list(range(1, 9))
Xp, npn = near_profile(g, MAN, dv, "man")
X = np.hstack([cnt[:, 1:], Xo, Xp, np.ones((n, 1))])
for nm, m in (("SHORTAGE d <= -2", d <= -2), ("SURPLUS d >= +2", d >= 2)):
    A = X[m]; keep = A.std(0) > 0; keep[-1] = True
    w = np.zeros(X.shape[1]); w[keep] = np.linalg.solve(A[:, keep].T @ A[:, keep] + 1e-3 * np.eye(keep.sum()), A[:, keep].T @ t[m])
    r = t[m] - A @ w
    print(f"\n===== {nm}: n={m.sum()} resid sd {r.std():.2f}")
    print("counts rel R:", dict(zip("ICPO", w[:4].round(2))))
    H = w[4:29].reshape(5, 5); V = w[29:54].reshape(5, 5)
    print("horizontal raw [left][right] (minus median):"); print((H - np.median(H)).round(1))
    print("vertical raw [upper][lower] (minus median):"); print((V - np.median(V)).round(1))
    P = w[54:54 + len(npn)].reshape(5, 5, len(dv))
    print("distance profile (d=2..8, none) relative to d=1 (only rows with |coef|>1.5):")
    for aa in range(5):
        for b in range(5):
            if np.abs(P[aa, b, :5]).max() > 1.5:
                print(f"  {SYM[aa]} -> nearest {SYM[b]}: " + " ".join(f"{v:+7.2f}" for v in P[aa, b]))
