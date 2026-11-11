"""Wellness / Transportation: combine directional adjacency with distance profiles under several metrics."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=2)
a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
g, y, z, lev, idx = labelled(a)
cen = (y > 0.08) & (y < 0.92)
t = lev.astype(float); ci = np.where(cen)[0]
C = counts(g)[:, 1:]
B = {}
B["ord4"] = ordered_pairs(g)[0]
B["ord8"] = ordered_pairs(g, ((1, 1), (1, -1)))[0]
B["prof_man"] = near_profile(g, MAN, list(range(1, 9)), "man")[0]
B["prof_che"] = near_profile(g, CHE, list(range(1, 7)), "che")[0]
B["prof_euc"] = near_profile(g, EUC2, sorted(set(EUC2.ravel()) - {0})[:14], "euc2")[0]
B["absent"] = absent(g)[0]
B["cnt2"] = np.stack([counts(g)[:, i] * counts(g)[:, j] for i in range(5) for j in range(i, 5)], 1)
print(f"===== {ADVISORS[a]} central n={len(ci)} sd(level) {t[ci].std():.1f}")
for combo in (["ord4"], ["prof_man"], ["prof_che"], ["prof_euc"], ["ord4", "prof_man"], ["ord4", "prof_che"], ["ord4", "prof_euc"], ["ord4", "ord8", "prof_man"],
              ["ord4", "prof_man", "cnt2"], ["ord4", "prof_man", "prof_euc"], ["ord4", "prof_man", "prof_che", "prof_euc", "cnt2"]):
    X = np.hstack([C] + [B[k] for k in combo])
    o, i, w = cv_fit(X, t, ci, lam=1e-2)
    print(f"  {'+'.join(combo):45s} cols {X.shape[1]:4d}  OOF resid sd {o:7.3f}  in-sample {i:7.3f}")
