"""Start set for the frozen-core test (Nov 12 evening): the valid grids of a constructed design first (the "core": any
two of them differ in >= 39 cells), then the invalid code grids closest to validity (the annealer's repair phase makes
those valid).   python explore/core_prep.py ckpt/crystal/c11.bin cache/core_c11.npy
Also writes cache/core_c11.k (the number of core grids, for --frozen / --special)."""
import sys, numpy as np
sys.path.insert(0, "src")
from rules import raw_scores, valid, THRESH
from divscore import int_score
SC = np.array([78.0, 56.0, 32.0, 160.0])            # same scales as deficit() in src/anneal.cpp
G = np.fromfile(sys.argv[1], np.uint8)[:125 * 49].reshape(125, 7, 7).astype(int)
raw, ok = raw_scores(G)
v = valid(G)
dfc = np.where(ok, (np.maximum(0, np.asarray(THRESH)[None] - raw) / SC[None]).sum(1), 50.0)
inv = np.flatnonzero(~v)
order = np.concatenate([np.flatnonzero(v), inv[np.argsort(dfc[inv], kind="stable")]])[:100]
k = min(int(v.sum()), 99)
S = G[order]
np.save(sys.argv[2], S.astype(np.uint8))
open(sys.argv[2][:-4] + ".k", "w").write("%d\n" % k)
D = (S.reshape(100, 1, 49) != S.reshape(1, 100, 49)).sum(2)
dc = D[:k, :k][np.triu_indices(k, 1)]
print(f"{sys.argv[1]}: core {k} valid grids (pairs at 39/40/49: {(dc == 39).sum()}/{(dc == 40).sum()}/{(dc == 49).sum()}); "
      f"score of the 100 code grids with validity ignored: {int_score(S)}")
print("deficits of the other grids:", np.round(dfc[order[k:]], 2).tolist())
