"""Where do the remaining close pairs sit?  Per-grid count of pairs < 38 vs freedom of the grid."""
import numpy as np, sys
sys.path.insert(0, "src")
from rules import raw_scores, valid, THRESH
from divscore import hamming_matrix
from anneal import read_state
np.set_printoptions(linewidth=220, suppress=True, precision=2)
src = sys.argv[1] if len(sys.argv) > 1 else "ckpt/g15/final.npz"
G = (read_state(src)["bestG"] if src.endswith(".bin") else np.load(src)["grids"]).reshape(100, 49)
D = hamming_matrix(G).astype(int); np.fill_diagonal(D, 99)
bad = (D < 38).sum(1)
raw, ok = raw_scores(G); mg = raw - THRESH[None]
cell = np.repeat(np.arange(49), 4)
fs = []
for i in range(100):
    g = G[i]; c = np.repeat(g[None], 196, 0); c[np.arange(196), cell] = (g[cell] + np.tile(np.arange(1, 5), 49)) % 5
    fs.append(valid(c).sum())
fs = np.array(fs)
print("pairs<38 per grid: histogram", np.bincount(bad), " total pairs", bad.sum() // 2)
print("corr(bad pairs, #valid single moves) =", np.corrcoef(bad, fs)[0, 1].round(3), " corr(bad, min margin/scale) =", np.corrcoef(bad, (mg / np.array([78, 56, 32, 160.])).min(1))[0, 1].round(3))
o = np.argsort(-bad)
print("worst grids: bad pairs, #valid single moves, margins W/T/Tr/B, counts")
for i in o[:12]:
    print(f"  grid {i:2d}: bad {bad[i]:2d}  valid singles {fs[i]:3d}  margins {mg[i].tolist()}  counts {np.bincount(G[i], minlength=5).tolist()}")
print("grids with 0 bad pairs:", int((bad == 0).sum()), " mean valid singles there", fs[bad == 0].mean().round(1), " vs with >=3 bad:", fs[bad >= 3].mean().round(1) if (bad >= 3).any() else None)
# structure of the bad-pair graph: connected components
import scipy.sparse.csgraph as cg
ncomp, lab = cg.connected_components(D < 38, directed=False)
sizes = np.bincount(lab)
print("bad-pair graph: components", ncomp, " sizes", np.sort(sizes)[::-1][:10])
