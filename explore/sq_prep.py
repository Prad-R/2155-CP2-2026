"""Start set for the special-row tests: the grids of a set reordered so that those with the fewest neighbours at
distance <= 38 come first (the first M grids are the special ones).  python explore/sq_prep.py <set.npy> <out.npy>"""
import sys, numpy as np
G = np.load(sys.argv[1]).reshape(100, 49)
D = (G[:, None, :] != G[None, :, :]).sum(2) + 1000 * np.eye(100, dtype=int)
k = (D <= 38).sum(1)
order = np.argsort(k, kind="stable")
np.save(sys.argv[2], G[order].reshape(100, 7, 7).astype(np.uint8))
print("neighbours at <=38, sorted:", k[order][:40].tolist(), "...")
