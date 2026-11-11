"""Grids whose label differs from the majority label of their raw value (hidden fine term): what do they share?"""
import numpy as np, sys
from lib import *
sys.path.insert(0, "src")
from rules import raw_scores
from dual import bus_raw
G, Y = data()
for a in (1, 3):
    idx = np.where(~np.isnan(Y[:, a]))[0]
    y = np.round(Y[idx, a], 7); g = G[idx]
    f = bus_raw(g) if a == 3 else raw_scores(g)[0][:, 1]
    dev = np.zeros(len(idx), int)
    for r in np.unique(f):
        m = f == r
        u, c = np.unique(y[m], return_counts=True)
        if len(u) > 1:
            maj = u[np.argmax(c)]
            dev[m & (y < maj)] = -1; dev[m & (y > maj)] = 1
    cnt = np.stack([(g == k).sum((1, 2)) for k in range(5)], 1)
    print(f"\n===== {ADVISORS[a]}: deviants below majority {int((dev == -1).sum())}, above {int((dev == 1).sum())} of {len(idx)}")
    print("  mean counts: deviants -1", cnt[dev == -1].mean(0).round(2) if (dev == -1).any() else None, " +1", cnt[dev == 1].mean(0).round(2) if (dev == 1).any() else None, " others", cnt[dev == 0].mean(0).round(2))
    # simple candidate properties
    props = {"has no R": cnt[:, 0] == 0, "no I": cnt[:, 1] == 0, "no C": cnt[:, 2] == 0, "no P": cnt[:, 3] == 0, "no O": cnt[:, 4] == 0,
             "nR odd": cnt[:, 0] % 2 == 1, "nI odd": cnt[:, 1] % 2 == 1, "nC odd": cnt[:, 2] % 2 == 1, "nP odd": cnt[:, 3] % 2 == 1, "nO odd": cnt[:, 4] % 2 == 1}
    for k, v in props.items():
        print(f"    {k:10s}: among below {v[dev == -1].mean() if (dev == -1).any() else float('nan'):.2f}  above {v[dev == 1].mean() if (dev == 1).any() else float('nan'):.2f}  others {v[dev == 0].mean():.2f}")
    ids = np.where(dev != 0)[0][:10]
    for s in range(0, len(ids), 10):
        blk = ids[s:s + 10]
        print("  " + "  ".join(f"{dev[i]:+d} f={f[i]:<6d}".ljust(11) for i in blk))
        print("  " + "  ".join(f"y={y[i]:.4f}".ljust(11) for i in blk))
        for r in range(7):
            print("  " + "  ".join(gstr(g[i])[r].ljust(11) for i in blk))
