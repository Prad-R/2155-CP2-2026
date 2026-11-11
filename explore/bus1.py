"""Business: block search.  Target: z (levels are not integer-spaced here), plus level for reference."""
import numpy as np, sys
from lib import *
from blocks import *

np.set_printoptions(linewidth=220, suppress=True, precision=3)
g, y, z, lev, idx = labelled(3)
n = len(g)
cnt = counts(g)
cen = (y > 0.08) & (y < 0.92)
ci = np.where(cen)[0]
Of = onehot(g)
B = {}
B["cnt"] = cnt[:, 1:]
B["cnt2"] = np.stack([cnt[:, i] * cnt[:, j] for i in range(5) for j in range(i, 5)], 1)
B["cntoh"] = np.concatenate([(cnt[:, k:k + 1] == np.arange(50)[None]).astype(np.float32) for k in range(5)], 1)
B["ord4"] = ordered_pairs(g)[0]
B["ord8"] = ordered_pairs(g, ((1, 1), (1, -1)))[0]
B["ord2"] = ordered_pairs(g, ((0, 2), (2, 0)))[0]
B["prof_man"] = near_profile(g, MAN, list(range(1, 9)), "man")[0]
B["prof_che"] = near_profile(g, CHE, list(range(1, 7)), "che")[0]
B["pos"] = Of.reshape(n, -1)[:, 49:]
for tname, t in (("level", lev.astype(float)), ("z", z * 100)):
    print(f"===== Business target {tname}: sd {t[ci].std():.2f}")
    for combo in (["cnt"], ["cnt", "cnt2"], ["cntoh"], ["cnt", "cnt2", "ord4"], ["cnt", "cnt2", "prof_man"], ["cnt", "cnt2", "ord4", "prof_man"], ["cnt", "cnt2", "ord4", "ord8"],
                  ["cnt", "cnt2", "ord4", "pos"], ["cnt", "cnt2", "ord4", "ord8", "ord2", "prof_man", "prof_che"]):
        X = np.hstack([B[k] for k in combo])
        o, i, w = cv_fit(X, t, ci, lam=1e-2)
        print(f"  {'+'.join(combo):50s} cols {X.shape[1]:4d}  OOF {o:8.3f}  in-sample {i:8.3f}")
