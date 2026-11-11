"""Exact-score confirmation: score of each submission with the rule-invalid grids removed vs the returned score."""
import numpy as np, json, sys
sys.path.insert(0, "src")
from rules import raw_scores, NAMES
from divscore import int_score, DENOM

thr = json.load(open("results/exact_thresholds.json"))
T = np.array([thr[nm] for nm in NAMES])
log = [json.loads(l) for l in open("results/submission_log.jsonl")]
allok = True
for d in log:
    S = np.load(d["file"])
    raw, ok = raw_scores(S)
    valid = (raw >= T[None]).all(1) & ok
    pred = int_score(S, valid)
    ret = d.get("returned_score")
    if ret is None:
        print(f"  {d['name']:4s} not submitted: rules predict {pred / DENOM:.10f} ({int((~valid).sum())} invalid)")
        continue
    match = pred == int(round(ret * DENOM))
    allok &= match
    print(f"  {d['name']:4s} returned {ret:.10f} (int {int(round(ret * DENOM))})  rules predict {pred / DENOM:.10f} (int {pred})  invalid {int((~valid).sum())}  {'EXACT MATCH' if match else 'MISMATCH'}")
print("all scored submissions reproduced exactly:", allok)
