"""EDA 1: label structure, dataset composition, top/bottom grids per advisor."""
import numpy as np, sys
sys.path.insert(0, "src")
from common import load_data, ADVISORS

np.set_printoptions(linewidth=200, suppress=True)
G, Y = load_data()
print("grids", G.shape, G.dtype, "scores", Y.shape)
cnt = np.stack([(G == t).sum((1, 2)) for t in range(5)], 1)
print("dataset count mean", cnt.mean(0).round(2), "std", cnt.std(0).round(2), "min", cnt.min(0), "max", cnt.max(0))
for t in range(5):
    print("type", t, "count hist(0..49 binned by 5):", np.histogram(cnt[:, t], bins=np.arange(0, 55, 5))[0])
# duplicates in the dataset?
flat = G.reshape(len(G), 49)
u = np.unique(flat, axis=0)
print("unique grids", len(u), "of", len(flat))
SYM = "RICPO"


def show(gs, labels=None, per_row=10):
    gs = np.asarray(gs)
    for s in range(0, len(gs), per_row):
        blk = gs[s:s + per_row]
        if labels is not None:
            print("  ".join(f"{l:<7}" for l in labels[s:s + per_row]))
        for r in range(7):
            print("  ".join("".join(SYM[v] for v in g[r]) for g in blk))
        print()


for a, name in enumerate(ADVISORS):
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = Y[idx, a]
    k = y * 999
    print(f"\n===== {name}: n={len(idx)} unique={len(np.unique(y))} on k/999 grid: {np.mean(np.abs(k - np.round(k)) < 1e-6):.4f}  on k/1998: {np.mean(np.abs(2 * k - np.round(2 * k)) < 1e-6):.4f}")
    vals, c = np.unique(y, return_counts=True)
    print("  largest tie groups:", sorted(zip(c, vals.round(6)), reverse=True)[:8])
    print("  smallest values:", vals[:6].round(6), c[:6], " largest:", vals[-6:].round(6), c[-6:])
    nongrid = y[np.abs(k - np.round(k)) > 1e-6]
    print("  off-grid values (first 10):", np.sort(nongrid)[:10], "count", len(nongrid))
    o = np.argsort(y)
    print("  counts of top 20:", cnt[idx[o[-20:]]].mean(0).round(1), " bottom 20:", cnt[idx[o[:20]]].mean(0).round(1))
    print("  BOTTOM 20 (R=res I=ind C=com P=park O=office)")
    show(G[idx[o[:20]]], [f"{v:.4f}" for v in y[o[:20]]])
    print("  TOP 20")
    show(G[idx[o[-20:]]], [f"{v:.4f}" for v in y[o[-20:]]])
