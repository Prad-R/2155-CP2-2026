"""List the best exact-validity sets found so far by the C++ chains (ckpt/<tag>/state.bin).

  python -u src/best_chains.py [--top 15] [--prefix p,q,n] [--verify]

Reads only the 32-byte header of each state file unless --verify is given (then the best set of the top
entries is re-checked with the Python rules and the exact score)."""
import argparse, glob, os, struct
import numpy as np

from common import ROOT

p = argparse.ArgumentParser()
p.add_argument("--top", type=int, default=15)
p.add_argument("--prefix", default="")
p.add_argument("--verify", action="store_true")
args = p.parse_args()
rows = []
for f in glob.glob(os.path.join(ROOT, "ckpt", "*", "state.bin")):
    tag = os.path.basename(os.path.dirname(f))
    if args.prefix and not any(tag.startswith(x) for x in args.prefix.split(",")):
        continue
    try:
        with open(f, "rb") as fh:
            magic, step, done, best = struct.unpack("<4q", fh.read(32))
    except Exception:
        continue
    if magic != 0x414e4e45414c31:
        continue
    rows.append((best, tag, step, done, os.path.getmtime(f)))
rows.sort(reverse=True)
print(f"{len(rows)} chains; best first")
for best, tag, step, done, mt in rows[:args.top]:
    line = f"  {tag:14s} best {best} ({best / 49000:.10f})  step {step / 1e6:7.1f}M  {'done' if done else 'running/stopped'}"
    if args.verify:
        from anneal import read_state
        from rules import valid
        from divscore import int_score
        st = read_state(os.path.join(ROOT, "ckpt", tag, "state.bin"))
        G = st["bestG"].astype(int)
        ok = bool(valid(G).all()) and int_score(G) == st["best"] and len(np.unique(G.reshape(100, -1), axis=0)) == 100
        line += f"  verified={'YES' if ok else 'NO'}"
    print(line)
