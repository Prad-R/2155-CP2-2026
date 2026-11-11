"""Residual grids of the counts+adjacency level model: look for the missing term."""
import numpy as np, sys
from lib import *
from tax1 import adj_pairs

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1])
g, y, z, lev, idx = labelled(a)
C = counts(g)
P1, n1 = adj_pairs(g)
cen = (y > 0.1) & (y < 0.9)
t = lev.astype(float)
X = np.hstack([C[:, 1:], P1, np.ones((len(g), 1))])
w, *_ = np.linalg.lstsq(X[cen], t[cen], rcond=None)
res = t - X @ w
print(ADVISORS[a], "resid sd central", res[cen].std())
ci = np.where(cen)[0]
o = ci[np.argsort(res[ci])]


def show(ids):
    for s in range(0, len(ids), 8):
        blk = ids[s:s + 8]
        print("  ".join(f"r{res[i]:+6.1f} L{lev[i]:<4d}" for i in blk))
        print("  ".join(" ".join(str(int(v)) for v in C[i]).ljust(12) for i in blk))
        for r in range(7):
            print("  ".join(gstr(g[i])[r].ljust(12) for i in blk))
        print()


print("MOST NEGATIVE residual (true lower than linear model)")
show(o[:16])
print("MOST POSITIVE residual")
show(o[-16:])
# residual vs simple stats
dm = near_dist(g, MAN)
for b in range(5):
    for aa in range(5):
        m = onehot(g)[:, aa]
        v = (dm[:, b] * m).sum(1)
        print(f"corr(resid, sum over {SYM[aa]} of dist to nearest {SYM[b]}) = {np.corrcoef(res[cen], v[cen])[0, 1]:+.3f}", end=" | ")
    print()
