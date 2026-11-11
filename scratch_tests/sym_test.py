"""Does re-orienting (rotate/flip) or shifting grids raise the set score?"""
import json, os, numpy as np
from common import ROOT, d4
from divscore import int_score, DENOM, hamming_matrix
from surrogate import CNNEns, GBMEns
from blend import blend_apply, p_valid
cfg = json.load(open(os.path.join(ROOT, "results", "blend_b0.json"))); B = cfg["blend"]
cnn, gbm = CNNEns(cfg["cnn"], cfg["augs"]), GBMEns(cfg["gbm"])
def pv(g):
    pred = blend_apply(B, gbm.predict(g, procs=8).mean(2), cnn.predict(g).mean(2)); return pred, p_valid(B, pred)[0]
rng = np.random.RandomState(0)
for t in ["b2", "x4"]:
    G = np.load(os.path.join(ROOT, "ckpt", t, "final.npz"))["grids"].reshape(100, 7, 7).astype(np.int8)
    s0 = int_score(G)
    # 1. D4 copies: how far is a grid from its own rotations, and model validity of them
    T = np.stack([np.stack(d4(g)) for g in G])                       # 100,8,7,7
    self_d = (T[:, 1:] != T[:, :1]).reshape(100, 7, -1).sum(2)
    _, p8 = pv(T.reshape(-1, 7, 7)); p8 = p8.reshape(100, 8)
    print(f"{t}: start {s0/DENOM:.4f}; grid vs its own 7 transforms: mean {self_d.mean():.1f} cells; transforms with P(valid)>=0.95: {(p8[:,1:]>=0.95).mean():.2f}, >=0.5: {(p8[:,1:]>=0.5).mean():.3f}, min {p8.min():.3f}")
    # 2. greedy re-orientation of each grid (validity unchanged if advisors are symmetric)
    cur, H = s0, G.copy()
    for sweep in range(6):
        ch = 0
        for i in rng.permutation(100):
            best, bg = cur, None
            for k in range(1, 8):
                H2 = H.copy(); H2[i] = d4(H[i])[k]; s = int_score(H2)
                if s > best: best, bg = s, H2[i].copy()
            if bg is not None: H[i] = bg; cur = best; ch += 1
        print(f"   re-orientation sweep {sweep}: {cur/DENOM:.4f} ({ch} changed)")
        if not ch: break
    np.save(os.path.join(ROOT, "scratch_tests", f"{t}_reoriented.npy"), H)
    # 3. random orientation of every grid
    r = [int_score(np.stack([d4(g)[rng.randint(8)] for g in G])) / DENOM for _ in range(20)]
    print(f"   random orientations: mean {np.mean(r):.4f} max {np.max(r):.4f}")
    # 4. cyclic shifts (wrap-around translation): do the models still call them valid?
    sh = np.stack([np.roll(G, (dr, dc), (1, 2)) for dr in range(7) for dc in range(7) if dr or dc])   # 48,100,7,7
    pred, ps = pv(sh.reshape(-1, 7, 7)); ps = ps.reshape(48, 100); pred = pred.reshape(48, 100, 4)
    print(f"   cyclic shifts: P(valid)>=0.95 share {(ps>=0.95).mean():.3f}, >=0.5 share {(ps>=0.5).mean():.3f}; mean pred W/T/Tr/B {pred.mean((0,1)).round(3)} vs original {pv(G)[0].mean(0).round(3)}")
