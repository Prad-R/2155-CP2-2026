import itertools, json, sys
import numpy as np
sys.path.insert(0, "src")
from divscore import int_score
for name, obs in (("S14", 36081), ("S15", 34595)):
    t = json.load(open(f"results/probes/{name}.json")); sub = np.load(t["file"]); n = len(t["risky_idx"])
    tab = {}
    for r in range(n + 1):
        for c in itertools.combinations(range(n), r):
            v = np.ones(100, bool); v[[100 - n + i for i in c]] = False
            tab.setdefault(int_score(sub, v), []).append(list(c))
    print(name, "all-valid", int_score(sub), "distinct", len(tab), "of", 2 ** n, "| observed", obs, "-> failed risky (0-based, ascending model P):", tab.get(obs), "P:", [round(x, 3) for x in t["risky_pvalid"]])
    t["table"] = {str(k): v for k, v in tab.items()}; t["note"] = "table recomputed 2026-11-11 with correct risky positions (100-n+i)"
    json.dump(t, open(f"results/probes/{name}.json", "w"))
