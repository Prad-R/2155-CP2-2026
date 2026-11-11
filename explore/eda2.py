"""EDA 2: raw levels and near-duplicate labelled pairs (what does one cell change do?)."""
import numpy as np, sys
sys.path.insert(0, "src")
from common import load_data, ADVISORS

np.set_printoptions(linewidth=220, suppress=True)
G, Y = load_data()
SYM = "RICPO"


def gstr(g):
    return ["".join(SYM[v] for v in r) for r in g]


for a, name in enumerate(ADVISORS):
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = Y[idx, a]
    g = G[idx].reshape(len(idx), 49)
    vals, inv, c = np.unique(y, return_inverse=True, return_counts=True)
    k = vals * 1998
    ongrid = np.abs(k - np.round(k)) < 1e-5
    print(f"\n===== {name}: levels={len(vals)} on-grid levels={ongrid.sum()} samples on-grid={ongrid[inv].mean():.3f}")
    # width of each tie level in landmark units vs count: count/5000*999 should ~ span
    # print central levels: value*999, count
    mid = (vals > 0.45) & (vals < 0.55)
    print("  central levels (label*999, count):", [(round(v * 999, 2), int(n)) for v, n in zip(vals[mid], c[mid])][:40])
    lo = vals < 0.012
    print("  low tail (label*999, count):", [(round(v * 999, 3), int(n)) for v, n in zip(vals[lo], c[lo])])
    hi = vals > 0.988
    print("  high tail (label*999, count):", [(round(v * 999, 3), int(n)) for v, n in zip(vals[hi], c[hi])])
    # near-duplicate pairs
    D = np.zeros((len(g), len(g)), np.int16)
    for s in range(0, len(g), 500):
        D[s:s + 500] = (g[s:s + 500, None, :] != g[None, :, :]).sum(2)
    np.fill_diagonal(D, 99)
    for d in range(0, 5):
        print(f"  pairs at Hamming distance {d}: {(np.triu(D == d, 1)).sum()}")
    nn = D.min(1)
    print("  nearest-neighbour distance histogram:", np.bincount(nn, minlength=50)[:50])
    I, J = np.where(np.triu(D <= 1, 1))
    print("  --- pairs at distance <=1 (level diff = level index difference) ---")
    for i, j in list(zip(I, J))[:60]:
        diff = np.where(g[i] != g[j])[0]
        desc = ", ".join(f"({p // 7},{p % 7}) {SYM[g[i][p]]}->{SYM[g[j][p]]}" for p in diff)
        print(f"   y {y[i]:.6f} -> {y[j]:.6f}  lev {inv[i]} -> {inv[j]} (d={inv[j] - inv[i]:+d})  {desc}")
        gi = gstr(g[i].reshape(7, 7))
        for r in range(7):
            print("      " + gi[r])
