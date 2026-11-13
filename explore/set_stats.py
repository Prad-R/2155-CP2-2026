"""Distance structure of a 100-grid set: python explore/set_stats.py submissions/S21_....npy"""
import sys, numpy as np
sys.path.insert(0, "src")
from rules import raw_scores, THRESH
G = np.load(sys.argv[1]).reshape(100, 49).astype(int)
D = (G[:, None, :] != G[None, :, :]).sum(2)
iu = np.triu_indices(100, 1)
d = D[iu]
print("pairs by distance:", {int(k): int((d == k).sum()) for k in np.unique(d)})
print("mean pairwise distance %.3f" % d.mean())
Dn = D + 1000 * np.eye(100, dtype=int)
srt = np.sort(Dn, 1)
print("score", int(srt[:, :10].sum()), " 10-NN sum per grid:", dict(zip(*[x.tolist() for x in np.unique(srt[:, :10].sum(1), return_counts=True)])))
for k in (37, 38, 39, 40):
    deg = (Dn == k).sum(1)
    print("neighbours at exactly %d per grid: min %d mean %.1f max %d ; histogram %s" % (k, deg.min(), deg.mean(), deg.max(), dict(zip(*[x.tolist() for x in np.unique(deg, return_counts=True)]))))
le = (Dn <= 38).sum(1)
print("grids with fewer than 10 neighbours at <=38:", int((le < 10).sum()))
cnt = np.stack([(G == s).sum(1) for s in range(5)], 1)
print("cells per type R I C P O: mean", cnt.mean(0).round(2).tolist(), "min", cnt.min(0).tolist(), "max", cnt.max(0).tolist())
p = np.stack([(G == s).mean(0) for s in range(5)], 1)          # 49 x 5
h = 1 - (p ** 2).sum(1)
print("per-cell diversity 1-sum p^2: min %.3f mean %.4f max %.3f (uniform = 0.800)" % (h.min(), h.mean(), h.max()))
print("upper bound on mean pairwise distance from the per-cell frequencies: %.3f (balanced cells: 39.596)" % (h.sum() * 100 / 99))
raw, ok = raw_scores(G.reshape(100, 7, 7))
m = raw - THRESH[None]
print("margin over threshold [W, T, Tr, B]: min", m.min(0).tolist(), "median", np.median(m, 0).tolist(), "; grids at margin 0-1 per advisor:", (m <= 1).sum(0).tolist())
print("park frequency by cell (x100):"); print((p[:, 3].reshape(7, 7) * 100).round().astype(int))
print("residential frequency by cell (x100):"); print((p[:, 0].reshape(7, 7) * 100).round().astype(int))
