"""What limits the constrained optimum?  Margins, composition, freedom of moves, distance distribution."""
import numpy as np, sys
sys.path.insert(0, "src")
from rules import raw_scores, valid, THRESH, NAMES
from divscore import hamming_matrix, int_score
np.set_printoptions(linewidth=200, suppress=True, precision=2)
SW1, SW2 = np.triu_indices(49, 1)
for nm, f in (("constrained g15", "ckpt/g15/final.npz"), ("unconstrained", "ckpt/g21_unc/final.npz"), ("S8", "submissions/S8_20261110_234101.npy")):
    G = np.load(f); G = (G["grids"] if f.endswith(".npz") else G).reshape(100, 49)
    raw, ok = raw_scores(G)
    mg = raw - THRESH[None]
    cnt = np.stack([(G == k).sum(1) for k in range(5)], 1)
    D = hamming_matrix(G).astype(int); iu = np.triu_indices(100, 1); d = D[iu]
    print(f"\n===== {nm}: score {int_score(G.reshape(100,7,7))}  valid {int(valid(G).sum())}")
    print("  counts mean", cnt.mean(0).round(2), "sd", cnt.std(0).round(2), "min", cnt.min(0), "max", cnt.max(0))
    for a in range(4):
        print(f"  {NAMES[a]:14s} margin: min {mg[:, a].min():5d} median {np.median(mg[:, a]):6.1f}  grids with margin 0..2: {int(((mg[:, a] >= 0) & (mg[:, a] <= 2)).sum()):3d}  <=10: {int(((mg[:, a] >= 0) & (mg[:, a] <= 10)).sum()):3d}")
    print(f"  pairwise distance: mean {d.mean():.3f} sd {d.std():.3f} min {d.min()} hist {dict(zip(*np.unique(d, return_counts=True)))}")
    f_pos = np.stack([(G == k).mean(0) for k in range(5)], 0)              # 5,49
    print(f"  mean pairwise distance implied by per-position type frequencies: {(49 - (f_pos ** 2).sum()) * 100 / 99:.3f}; with uniform types 39.596")
    print("  per-position frequency of each type (x100), rows of the grid:")
    for k in range(5):
        print(f"    {'RICPO'[k]}: " + " | ".join(" ".join(f"{int(round(100 * v)):2d}" for v in f_pos[k].reshape(7, 7)[r]) for r in range(7)))
    if nm.startswith("constrained") or nm == "S8":
        fs, fw = [], []
        for i in range(100):
            g = G[i]
            c = np.repeat(g[None], 196, 0); cell = np.repeat(np.arange(49), 4); c[np.arange(196), cell] = (g[cell] + np.tile(np.arange(1, 5), 49)) % 5
            fs.append(valid(c).mean())
            m = g[SW1] != g[SW2]; s = np.repeat(g[None], m.sum(), 0); a, b = SW1[m], SW2[m]; s[np.arange(len(a)), a] = g[b]; s[np.arange(len(a)), b] = g[a]
            fw.append(valid(s).mean())
        print(f"  fraction of valid neighbours per grid: single-cell changes mean {np.mean(fs):.3f} (min {np.min(fs):.3f}); two-cell swaps mean {np.mean(fw):.3f} (min {np.min(fw):.3f})")
