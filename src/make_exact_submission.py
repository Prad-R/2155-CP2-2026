"""Package a 100-grid set whose validity is certified by the exact rules (src/rules.py) and log it.

  python -u src/make_exact_submission.py --name S18 --source ckpt/g7/state.npz --purpose "..."

--source: .npy (100,7,7), or .npz with `grids`, or an optimiser state.npz / C++ state.bin (uses `bestG`).
Refuses to write unless all 100 grids are distinct and valid under rules.THRESH (the safe thresholds).
"""
import argparse, hashlib, json, os, time
import numpy as np

from common import ROOT
from rules import raw_scores, THRESH, THRESH_LABEL, NAMES
from divscore import int_score, DENOM

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--source", required=True)
p.add_argument("--purpose", default="")
args = p.parse_args()

if args.source.endswith(".bin"):  # C++ annealer state (ckpt/<tag>/state.bin, chain may still be running): best set so far
    from anneal import read_state
    G = read_state(os.path.join(ROOT, args.source))["bestG"]
else:
    z = np.load(os.path.join(ROOT, args.source))
    G = (z["bestG"] if "bestG" in z.files else z["grids"]) if args.source.endswith(".npz") else z
sub = np.asarray(G).reshape(100, 7, 7).astype(int)
assert sub.shape == (100, 7, 7) and sub.min() >= 0 and sub.max() <= 4
assert len(np.unique(sub.reshape(100, -1), axis=0)) == 100, "duplicate grids"
raw, ok = raw_scores(sub)
marg = raw - THRESH[None]
assert ok.all() and (marg >= 0).all(), f"not all valid under the safe thresholds: {int((~(ok & (marg >= 0).all(1))).sum())} invalid"
S = int_score(sub)
ts = time.strftime("%Y%m%d_%H%M%S")
fn = os.path.join("submissions", f"{args.name}_{ts}.npy")
np.save(os.path.join(ROOT, fn), sub)
chk = np.load(os.path.join(ROOT, fn))
assert chk.shape == (100, 7, 7) and np.issubdtype(chk.dtype, np.integer) and (chk == sub).all()
h = hashlib.sha1(sub.astype(np.int8).tobytes()).hexdigest()[:8]
entry = dict(name=args.name, file=fn, sha=h, time=ts, source=args.source, pmin=1.0, pool=100,
             expected_int=S, expected_score=round(S / DENOM, 10), expected_valid=100.0, min_pvalid=1.0,
             exact_rules=True, thresholds=THRESH.tolist(),
             min_margin_over_label_threshold=dict(zip(NAMES, (raw - THRESH_LABEL[None]).min(0).tolist())),
             purpose=args.purpose, returned_score=None, decode=None)
with open(os.path.join(ROOT, "results", "submission_log.jsonl"), "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"saved {fn} sha={h}  all-valid score={S/DENOM:.10f} (int {S})  min margins over label thresholds {entry['min_margin_over_label_threshold']}", flush=True)
