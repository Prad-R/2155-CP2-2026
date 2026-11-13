"""Check the output of the construction probe with the Python rules: python explore/crystal_check.py ckpt/crystal/c*.bin"""
import sys, numpy as np
sys.path.insert(0, "src")
from rules import raw_scores, valid, THRESH
for path in sys.argv[1:]:
    b = np.fromfile(path, np.uint8)
    G = b[:125 * 49].reshape(125, 7, 7).astype(int)
    v = valid(G)
    D = (G.reshape(125, 1, 49) != G.reshape(1, 125, 49)).sum(2)
    d = D[np.triu_indices(125, 1)]
    raw, ok = raw_scores(G)
    fail = ((raw < THRESH[None]) | ~ok[:, None]).sum(0).tolist()
    cnt = np.stack([(G.reshape(125, 49) == s).sum(1) for s in range(5)], 1)
    print("%s: valid %d of 125 (Python rules); min distance among all 125: %d; failing [W, T, Tr, B] %s; parks per grid mean %.1f, among valid %s"
          % (path, int(v.sum()), int(d.min()), fail, cnt[:, 3].mean(), ("%.1f" % cnt[v, 3].mean()) if v.any() else "-"))
