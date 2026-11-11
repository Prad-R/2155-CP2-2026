"""Which cells can move?  Validity of single-cell changes and swaps by district type (final constrained set)."""
import numpy as np, sys
sys.path.insert(0, "src")
from rules import raw_scores, valid, THRESH
from anneal import read_state
np.set_printoptions(linewidth=220, suppress=True, precision=3)
src = sys.argv[1] if len(sys.argv) > 1 else "ckpt/g15/final.npz"
G = (read_state(src)["bestG"] if src.endswith(".bin") else np.load(src)["grids"]).reshape(100, 49).astype(int)
SYM = "RICPO"
cell = np.repeat(np.arange(49), 4); sh = np.tile(np.arange(1, 5), 49)
tot = np.zeros((5, 5)); ok = np.zeros((5, 5)); frozen = np.zeros(5); ncell = np.zeros(5)
SW1, SW2 = np.triu_indices(49, 1)
stot = np.zeros((5, 5)); sok = np.zeros((5, 5))
for i in range(100):
    g = G[i]
    c = np.repeat(g[None], 196, 0); new = (g[cell] + sh) % 5; c[np.arange(196), cell] = new
    v = valid(c)
    np.add.at(tot, (g[cell], new), 1); np.add.at(ok, (g[cell], new), v)
    per_cell = v.reshape(49, 4).sum(1)
    for t in range(5):
        ncell[t] += (g == t).sum(); frozen[t] += ((g == t) & (per_cell == 0)).sum()
    m = g[SW1] != g[SW2]; a, b = SW1[m], SW2[m]
    s = np.repeat(g[None], len(a), 0); s[np.arange(len(a)), a] = g[b]; s[np.arange(len(a)), b] = g[a]
    vs = valid(s)
    t1, t2 = np.minimum(g[a], g[b]), np.maximum(g[a], g[b])
    np.add.at(stot, (t1, t2), 1); np.add.at(sok, (t1, t2), vs)
print("single-cell change: fraction valid  [rows = current type, cols = new type]")
print("      " + "      ".join(SYM))
for t in range(5):
    print(f"  {SYM[t]}  " + "  ".join(f"{ok[t, u] / tot[t, u]:5.2f}" if tot[t, u] else "   - " for u in range(5)))
print("cells with no valid single change (frozen), by type:", {SYM[t]: f"{frozen[t] / ncell[t]:.2f}" for t in range(5)})
print("two-cell swap: fraction valid by type pair")
for t in range(5):
    print(f"  {SYM[t]}  " + "  ".join(f"{sok[t, u] / stot[t, u]:5.2f}" if stot[t, u] else "   - " for u in range(5)))
