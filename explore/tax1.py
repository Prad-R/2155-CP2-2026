"""Interpretable linear fits on the integer level index (central region) for each advisor."""
import numpy as np, sys
from lib import *

np.set_printoptions(linewidth=220, suppress=True, precision=3)
which = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1]


def adj_pairs(g, offs=((0, 1), (1, 0))):
    """unordered type-pair counts over the given offsets: (n,15)"""
    n = len(g)
    M = np.zeros((n, 5, 5), np.float32)
    for dr, dc in offs:
        a = g[:, :7 - dr if dr >= 0 else 7, max(0, -dc):7 - max(0, dc)] if dr >= 0 else None
        A = g[:, max(0, -dr):7 - max(0, dr), max(0, -dc):7 - max(0, dc)]
        B = g[:, max(0, dr):7 + min(0, dr) if dr < 0 else 7, :][:, :, max(0, dc):7 + min(0, dc) if dc < 0 else 7]
        B = g[:, max(0, dr):, max(0, dc):][:, :A.shape[1], :A.shape[2]] if dc >= 0 else g[:, max(0, dr):, :7 + dc][:, :A.shape[1], :]
        for s in range(5):
            for t in range(5):
                M[:, s, t] += ((A == s) & (B == t)).sum((1, 2))
    M = M + M.transpose(0, 2, 1)
    iu = np.triu_indices(5)
    out = M[:, iu[0], iu[1]].copy()
    out[:, iu[0] == iu[1]] /= 2
    names = [f"{SYM[i]}{SYM[j]}" for i, j in zip(*iu)]
    return out, names


for a in which:
    g, y, z, lev, idx = labelled(a)
    C = counts(g)
    P1, n1 = adj_pairs(g)                                   # 4-neighbour pairs
    Pd, nd = adj_pairs(g, offs=((1, 1), (1, -1)))           # diagonal pairs
    # central region: levels whose label in [0.1, 0.9]
    cen = (y > 0.1) & (y < 0.9)
    # level index as integer target (consecutive in the centre)
    t = lev.astype(float)
    print(f"\n===== {ADVISORS[a]}: central n={cen.sum()} level range {lev[cen].min()}..{lev[cen].max()}  sd(level)={t[cen].std():.1f}")
    for nm, X, names in (("counts", C[:, 1:], [f"n{SYM[i]}" for i in range(1, 5)]),
                         ("counts+adj", np.hstack([C[:, 1:], P1]), [f"n{SYM[i]}" for i in range(1, 5)] + n1),
                         ("counts+adj+diag", np.hstack([C[:, 1:], P1, Pd]), [f"n{SYM[i]}" for i in range(1, 5)] + n1 + ["d" + s for s in nd])):
        Xc = np.hstack([X[cen], np.ones((cen.sum(), 1))])
        w, *_ = np.linalg.lstsq(Xc, t[cen], rcond=None)
        res = t[cen] - Xc @ w
        print(f"  {nm}: resid sd {res.std():.2f} levels  R2 {1 - res.var() / t[cen].var():.4f}")
        print("    " + "  ".join(f"{n}:{v:+.2f}" for n, v in zip(names + ["const"], w)))
