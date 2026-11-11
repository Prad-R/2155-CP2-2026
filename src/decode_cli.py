"""Decode a returned leaderboard score for a logged submission and record it.
  python src/decode_cli.py NAME SCORE [max_invalid]
"""
import json, os, sys
import numpy as np
from common import ROOT
from divscore import decode, DENOM
name, sc = sys.argv[1], float(sys.argv[2]); mi = int(sys.argv[3]) if len(sys.argv) > 3 else 2
lp = os.path.join(ROOT, "results", "submission_log.jsonl")
rows = [json.loads(l) for l in open(lp)]
e = [r for r in rows if r["name"] == name][-1]
g = np.load(os.path.join(ROOT, e["file"]))
r = decode(g, sc, max_invalid=mi, p_valid=e.get("pvalid"))
print("target", r["target"], "all-valid", r["all_valid"], "deficit", r["all_valid"] - r["target"], "frac residual", sc * DENOM - r["target"])
print("n solutions", len(r["solutions"]))
for s, l in r["solutions"][:15]:
    print("  invalid idx", s, "pvalid", [e["pvalid"][i] for i in s], "dataset ids", [e["dataset_ids"][i] for i in s])
e["returned_score"] = sc
e["decode"] = dict(target=r["target"], solutions=[list(s) for s, _ in r["solutions"][:50]])
tmp = lp + ".tmp"
open(tmp, "w").write("".join(json.dumps(x) + "\n" for x in rows)); os.replace(tmp, lp)
