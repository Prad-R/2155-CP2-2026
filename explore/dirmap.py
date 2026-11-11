"""Fully ordered pair map: level ~ counts + sum over (type a at cell, type b at cell+(dr,dc)) coef.
Prints, per offset, the double-centred 5x5 coefficient matrix (pure interaction part) where it is significant."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=1)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
R = int(sys.argv[2]) if len(sys.argv) > 2 else 2
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float); n = len(g)
O = onehot(g).reshape(n, 5, 7, 7)
offs = [(dr, dc) for dr in range(0, R + 1) for dc in range(-R, R + 1) if (dr > 0 or dc > 0)]
cols = []
for dr, dc in offs:
    A = O[:, :, :7 - dr, max(0, -dc):7 - max(0, dc)]          # cell
    B = O[:, :, dr:, max(0, dc):7 + min(0, dc)]               # cell + (dr,dc)   (dr>0: below; dc>0: right)
    cols.append(np.einsum("nahw,nbhw->nab", A, B).reshape(n, 25))
C = counts(g)
# position-by-type indicators absorb all border/gauge effects
POSF = onehot(g)[:, 1:].reshape(n, -1) if len(sys.argv) > 4 else np.zeros((n, 0))
X = np.hstack([C[:, 1:]] + cols + [POSF, np.ones((n, 1))])
ci = np.where(cen)[0]; f = folds_for(len(ci), 5, 0)
lam = 1e-4
oof = np.zeros(len(ci))
for k in range(5):
    Atr = X[ci[f != k]]
    w = np.linalg.solve(Atr.T @ Atr + lam * np.eye(X.shape[1]), Atr.T @ t[ci[f != k]])
    oof[f == k] = X[ci[f == k]] @ w
Atr = X[ci]
w = np.linalg.solve(Atr.T @ Atr + lam * np.eye(X.shape[1]), Atr.T @ t[ci])
print(f"{ADVISORS[a]}: ordered pair map radius {R}: {X.shape[1]} cols, OOF resid sd {(t[ci] - oof).std():.3f}, in-sample {(t[ci] - Atr @ w).std():.3f}")
print("counts (rel. R):", dict(zip("ICPO", w[:4].round(2))))
for i, (dr, dc) in enumerate(offs):
    W = w[4 + 25 * i:4 + 25 * (i + 1)].reshape(5, 5)
    Wc = W - W.mean(1, keepdims=True) - W.mean(0, keepdims=True) + W.mean()
    if np.abs(Wc).max() > thr:
        print(f"offset (down {dr}, right {dc:+d}): rows = type at cell, cols = type at cell+offset  [double-centred]  max|.|={np.abs(Wc).max():.2f}")
        for r in range(5):
            print("    " + SYM[r] + " " + " ".join(f"{v:+6.2f}" for v in Wc[r]))
