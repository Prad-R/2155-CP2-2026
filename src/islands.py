"""Island chain: several annealing stages in a row, each later stage restarting from a best set.

  python -u src/islands.py TAG INIT RULE SEED  NxSTEPS <anneal args> [:: NxSTEPS <anneal args> ...]
  e.g. python -u src/islands.py y001 ckpt/v073/state.bin best 3001 6x50000000 --t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --dmin 39

Stage k of island TAG is the ordinary chain ckpt/<TAG>s<k>/ run by src/anneal.py (resumable, verified at the end).
Stage 1 starts from INIT.  A later stage starts from
  own  : the best set of this island's previous stage;
  best : the best set held at that moment by any chain whose tag has the same letters as TAG (every island of the
         run, including stages still running); ties are spread over the islands by island number.
A stage can name its own rule in front of its count: "own:1x20000000" or "best:3x20000000" (default: RULE).
Requeue-safe: finished stages are skipped (DONE marker), a stopped stage resumes from its checkpoint, and the start
set of a stage is fixed once the stage has started (anneal.py keeps init.bin).  SIGTERM/SIGUSR1 go to the stage.
"""
import os, sys, glob, signal, struct, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def header_best(path):
    """Best score stored in a chain's state.bin header, or -1 if it cannot be read."""
    try:
        with open(path, "rb") as f:
            return struct.unpack("<4q", f.read(32))[3]
    except Exception:
        return -1


def main():
    tag, init, rule, seed = sys.argv[1:5]
    assert rule in ("own", "best"), rule
    stages, group = [], []
    for a in sys.argv[5:] + ["::"]:
        if a != "::":
            group.append(a)
            continue
        if group:
            srule, _, spec = group[0].rpartition(":")  # a stage may name its own rule: own:1x20000000, best:2x20000000
            srule = srule or rule
            assert srule in ("own", "best"), srule
            n, steps = spec.split("x")
            stages += [(int(steps), group[1:], srule)] * int(n)
        group = []
    assert stages, "no stages given"
    letters = tag.rstrip("0123456789")
    num = int(tag[len(letters):])
    cur = {}

    def forward(*_):
        cur["stop"] = True
        p = cur.get("proc")
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGTERM)

    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, forward)

    for k, (steps, args, srule) in enumerate(stages, 1):
        st = f"{tag}s{k}"
        d = os.path.join(ROOT, "ckpt", st)
        if os.path.exists(os.path.join(d, "DONE")):
            continue
        if cur.get("stop"):
            break
        started = os.path.exists(os.path.join(d, "init.bin")) or os.path.exists(os.path.join(d, "state.bin"))
        if k == 1 or started:
            src = init  # anneal.py ignores --init once the stage has started
        elif srule == "own":
            src = f"ckpt/{tag}s{k - 1}/state.bin"
        else:
            cands = [(header_best(p), p) for p in sorted(glob.glob(os.path.join(ROOT, "ckpt", letters + "[0-9]*", "state.bin")))]
            top = max(b for b, _ in cands)
            ties = [p for b, p in cands if b == top]
            src = os.path.relpath(ties[num % len(ties)], ROOT)
        if started:
            print(f"=== island {tag} stage {k}/{len(stages)}: resuming", flush=True)
        else:
            b = header_best(os.path.join(ROOT, src)) if src.endswith(".bin") else "n/a"
            print(f"=== island {tag} stage {k}/{len(stages)}: start set {src} (best {b})", flush=True)
        cmd = [sys.executable, "-u", os.path.join(ROOT, "src", "anneal.py"), "--tag", st, "--init", src,
               "--steps", str(steps)] + args + ["--seed", str(int(seed) * 10 + k)]
        for opt, val in (("--log-every", "10000000"), ("--ckpt-every", "2000000")):
            if opt not in args:
                cmd += [opt, val]
        cur["proc"] = subprocess.Popen(cmd)
        rc = cur["proc"].wait()
        if rc != 0:
            print(f"=== island {tag} stage {k}: failed with code {rc}", flush=True)
            raise SystemExit(rc)
        if not os.path.exists(os.path.join(d, "DONE")):
            print(f"=== island {tag} stage {k}: stopped before the end (resumable)", flush=True)
            raise SystemExit(0)
    else:
        last = os.path.join(ROOT, "ckpt", f"{tag}s{len(stages)}", "state.bin")
        print(f"=== island {tag}: all {len(stages)} stages done, best {header_best(last)}", flush=True)


if __name__ == "__main__":
    main()
