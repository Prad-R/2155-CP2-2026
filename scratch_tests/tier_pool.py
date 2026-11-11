import numpy as np, sys
sys.path.insert(0, "src")
from divscore import hamming_matrix
u = np.load("ckpt/u1/final.npz"); print("u1 int", int(u["score"]), "leader int", round(0.7752244898 * 49000), "u1 pvalid quantiles", np.round(np.quantile(u["pvalid"], [0, .25, .5, .75, 1]), 4))
G, P, src = [], [], []
for t in ["w1", "w2", "w3", "w4", "w5", "w6", "w8", "u1"]:
    z = np.load(f"ckpt/{t}/final.npz"); G.append(z["grids"].reshape(-1, 49)); P.append(z["pvalid"]); src += [t] * len(z["pvalid"])
G, P, src = np.concatenate(G), np.concatenate(P), np.array(src)
core = np.load("submissions/S8_20261110_234101.npy").reshape(100, 49)
targets = [0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005]
pick = []
for tg in targets:
    o = np.argsort(np.abs(np.log(P + 1e-9) - np.log(tg)))
    for i in o:
        if i not in pick and hamming_matrix(G[i:i+1], core).min() > 0 and all(src[i] != src[j] or True for j in pick):
            pick.append(int(i)); break
print("picked b0 P:", np.round(P[pick], 4).tolist(), "from", src[pick].tolist())
np.savez("cache/probe_tier.npz", grids=G[pick].reshape(-1, 7, 7), pvalid=P[pick])
