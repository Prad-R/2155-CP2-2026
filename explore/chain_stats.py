"""Close-pair structure of the CURRENT state of chains: python explore/chain_stats.py <label>=<tag,tag,...> ...
For each group: mean over chains of pairs at 37 / 38 / 39 / 40, mean distance, 38-neighbours per grid, grids with < 10 neighbours at <= 38."""
import sys, numpy as np
sys.path.insert(0, "src")
from anneal import read_state


def stats(G):
    G = G.reshape(100, 49).astype(np.int16)
    D = (G[:, None, :] != G[None, :, :]).sum(2)
    d = D[np.triu_indices(100, 1)]
    Dn = D + 1000 * np.eye(100, dtype=int)
    k38 = (Dn == 38).sum(1)
    le = (Dn <= 38).sum(1)
    sc = np.sort(Dn, 1)[:, :10].sum()
    return [(d <= 36).sum(), (d == 37).sum(), (d == 38).sum(), (d == 39).sum(), (d == 40).sum(), (d >= 41).sum(), d.mean(), d.var(), k38.min(), (le < 10).sum(), sc]


print("%-34s %3s %5s %5s %6s %6s %6s %6s %7s %5s %6s %6s %7s" % ("group", "n", "<=36", "37", "38", "39", "40", ">=41", "mean", "var", "mink38", "n<10", "score"))
for arg in sys.argv[1:]:
    label, tags = arg.split("=")
    for which in ("G", "bestG"):
        rows = []
        for t in tags.split(","):
            try:
                rows.append(stats(read_state("ckpt/%s/state.bin" % t)[which]))
            except Exception as e:
                pass
        if not rows:
            continue
        m = np.mean(np.array(rows, float), 0)
        print("%-34s %3d %5.1f %5.1f %6.0f %6.0f %6.0f %6.0f %7.3f %5.2f %6.1f %6.1f %7.0f" % ((label + (" [current]" if which == "G" else " [best]"), len(rows)) + tuple(m)))
