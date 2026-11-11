import numpy as np, sys, json
sys.path.insert(0, "src")
from bus_model import BusEns
from rules import raw_scores, NAMES
bus = BusEns(dev="cpu")
thr = json.load(open("results/exact_thresholds.json")); T = np.array([thr[n] for n in NAMES])
for nm, f in (("S2", "submissions/S2_20261109_171634.npy"), ("S5", "submissions/S5_20261109_193339.npy"), ("S8", "submissions/S8_20261110_234101.npy"), ("S13", "submissions/S13_20261111_090940.npy"), ("u1", "ckpt/u1/final.npz"), ("e_test", "ckpt/e_test_quick/final.npz")):
    G = np.load(f); G = G["grids"] if f.endswith(".npz") else G
    m, s = bus.predict(G)
    raw, ok = raw_scores(G)
    mg = raw - T[None]
    print(f"{nm:7s} Business predicted level - {bus.thr}: quantiles {np.quantile(m - bus.thr, [0, .05, .25, .5, .75, .95, 1]).round(1)}  ens sd mean {s.mean():.1f} | "
          f"raw margins median W {np.median(mg[:,0]):.0f} T {np.median(mg[:,1]):.0f} Tr {np.median(mg[:,2]):.0f} | valid3 {(ok & (mg >= 0).all(1)).sum()}")
