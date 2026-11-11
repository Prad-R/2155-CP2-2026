"""Wrapper around the C++ annealer (src/anneal.cpp -> bin/anneal): exact-validity set optimisation.

  python -u src/anneal.py --tag h1 --init submissions/S8_20261110_234101.npy --steps 20000000 [--t0 1 --t1 0.05 --nswap 200 ...]

Resumable (ckpt/<tag>/state.bin), forwards SIGTERM/SIGUSR1 to the binary (which checkpoints and exits), and on
completion writes ckpt/<tag>/final.npz after re-verifying every grid with the Python rules and the exact score.
Extra arguments are passed to the binary unchanged (--t0 --t1 --nswap (-1 = all swaps) --focus --lam --dmin --lam2 --dmin2 --lam-end --lam2-end --cycles
--seed --log-every --ckpt-every --max-tries).   `python src/anneal.py selftest` cross-checks the C++ rules.
"""
import os, sys, signal, subprocess, struct
import numpy as np

from common import ROOT
from rules import raw_scores, valid, THRESH
from divscore import int_score, DENOM

BIN = os.environ.get("ANNEAL_BIN") or next(os.path.join(ROOT, "bin", b) for b in ("anneal3", "anneal2", "anneal") if os.path.exists(os.path.join(ROOT, "bin", b)))
MAGIC = 0x414e4e45414c31


def read_state(path):
    b = open(path, "rb").read()
    magic, step, done, best = struct.unpack("<4q", b[:32])
    assert magic == MAGIC
    G = np.frombuffer(b[32:32 + 4900], np.uint8).reshape(100, 7, 7).copy()
    Gb = np.frombuffer(b[32 + 4900:32 + 9800], np.uint8).reshape(100, 7, 7).copy()
    return dict(step=step, done=bool(done), best=best, G=G, bestG=Gb)


def load_set(path):
    path = os.path.join(ROOT, path)
    if path.endswith(".bin"):
        return read_state(path)["bestG"]
    z = np.load(path)
    if path.endswith(".npz"):
        z = z["bestG"] if "bestG" in z.files else z["grids"]
    return np.asarray(z).reshape(100, 7, 7).astype(np.uint8)


def selftest(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    g = rng.randint(0, 5, size=(n, 49)).astype(np.uint8)
    # include structured grids: copies of valid sets
    extra = [np.load(os.path.join(ROOT, "submissions", f)).reshape(-1, 49).astype(np.uint8) for f in sorted(os.listdir(os.path.join(ROOT, "submissions"))) if f.endswith(".npy")]
    g = np.concatenate([g] + extra)
    tmp = os.path.join(ROOT, "cache", f"selftest_{os.getpid()}.bin")
    g.tofile(tmp)
    out = subprocess.run([BIN, "--score", tmp], capture_output=True, text=True, check=True).stdout
    os.remove(tmp)
    c = np.array([[int(x) for x in l.split()] for l in out.strip().split("\n")])
    raw, ok = raw_scores(g)
    assert (c[:, 0] == ok).all(), "ok flag mismatch"
    assert (c[ok, 1:] == raw[ok]).all(), "raw score mismatch"
    print(f"selftest passed: {len(g)} grids ({int(ok.sum())} with all types), C++ raw scores identical to rules.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest(); raise SystemExit
    args = sys.argv[1:]
    tag = args[args.index("--tag") + 1]
    init = args[args.index("--init") + 1]
    rest = [a for i, a in enumerate(args) if a not in ("--tag", "--init") and (i == 0 or args[i - 1] not in ("--tag", "--init"))]
    out = os.path.join(ROOT, "ckpt", tag)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE")):
        print("already done"); raise SystemExit
    state, initb = os.path.join(out, "state.bin"), os.path.join(out, "init.bin")
    if not os.path.exists(state) and not os.path.exists(initb):
        load_set(init).tofile(initb + ".tmp"); os.replace(initb + ".tmp", initb)
    cmd = [BIN, "--init", initb, "--state", state, "--thr"] + [str(int(t)) for t in THRESH] + rest
    print("exec:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, lambda *a: proc.send_signal(signal.SIGTERM))
    rc = proc.wait()
    if rc != 0:
        print(f"annealer exited with code {rc} (99 = stopped by signal, resumable)", flush=True)
        raise SystemExit(0 if rc == 99 else rc)
    st = read_state(state)
    Gf = st["bestG"].astype(int)
    v = valid(Gf)
    sc = int_score(Gf)
    assert v.all(), f"{int((~v).sum())} grids fail the Python rules"
    assert sc == st["best"], (sc, st["best"])
    assert len(np.unique(Gf.reshape(100, -1), axis=0)) == 100
    raw, _ = raw_scores(Gf)
    tmp = os.path.join(out, "final.tmp.npz")
    np.savez(tmp, grids=Gf, raw=raw, score=sc)
    os.replace(tmp, os.path.join(out, "final.npz"))
    print(f"final score {sc / DENOM:.10f} (int {sc}); all 100 grids valid under the Python rules; min margins {(raw - THRESH[None]).min(0).tolist()}", flush=True)
    open(os.path.join(out, "DONE"), "w").write("ok\n")
