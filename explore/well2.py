"""Print the coefficients of counts + ordered N4 pairs + Manhattan nearest-distance profile."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
selfz = len(sys.argv) > 2 and sys.argv[2] == "self0"
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float); ci = np.where(cen)[0]
C = counts(g)[:, 1:]
Xo, no = ordered_pairs(g)
dv = list(range(1, 9))
Xp, npn = near_profile(g, MAN, dv, "man", self_zero=selfz)
X = np.hstack([C, Xo, Xp])
o, i, w = cv_fit(X, t, ci, lam=1e-2)
print(f"{ADVISORS[a]}: OOF {o:.3f} in-sample {i:.3f}")
print("counts rel R:", dict(zip("ICPO", w[:4].round(2))))
H = w[4:29].reshape(5, 5); V = w[29:54].reshape(5, 5)
dc = lambda W: W - W.mean(1, keepdims=True) - W.mean(0, keepdims=True) + W.mean()
print("horizontal [left][right], double-centred:"); print(dc(H).round(2))
print("vertical [upper][lower], double-centred:"); print(dc(V).round(2))
print("horizontal raw:"); print(H.round(2))
print("vertical raw:"); print(V.round(2))
P = w[54:54 + len(npn)].reshape(5, 5, len(dv))
print("distance profile (d=2..8, none) relative to d=1:")
for aa in range(5):
    for b in range(5):
        print(f"  {SYM[aa]} -> nearest {SYM[b]}: " + " ".join(f"{v:+7.2f}" for v in P[aa, b]))
