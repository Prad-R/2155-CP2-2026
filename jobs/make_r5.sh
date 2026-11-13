#!/bin/bash
# Writes jobs/opt_r5.txt: 96 island chains (src/islands.py, binary bin/anneal6), 240M steps each (~9 h at 7.5k
# steps/s). Usage: [HOT=n] bash jobs/make_r5.sh    Line n of the list has the log logs/cp2-optR5-<jobid>-<n>.out.
# HOT = number of hot stages (default 9); each one fewer shortens the run by 20M steps (~45 min): HOT=4 is ~5.2 h.
# Design (from the fixed-core chains kb01..kb14 and the temperature scan kt01..kt26, UPDATES.md 2026-11-12 23:50):
#   A start set of explore/core_prep.py = K valid code grids (kept fixed with --frozen K) + the other code grids.
#   From that start a chain at constant T 1.5-2.0 climbs steadily (still rising after 2M steps),
#   T 1.3 is slower, T >= 2.5 destroys the set, a free core is ~50 lower, no pair penalty ~130 lower.
#   Every island: 9 hot stages of 20M steps at its own (T, lam), then 40M steps cooling to T 0.8, then cold polish
#   of its own result with the core fixed (10M) and with the core free (10M).
#   W islands ("winners", 60): every hot stage and the cooling stage restart from the best set of the island's
#     group at that moment, so the group keeps working on its record.
#   O islands (18): restart from their own best only (independent lines).
#   C islands (18): one slow cooling stage of 180M steps (T 2.0 -> 1.0), then as above.
#   Groups (one per core, tags za/zb/zc/zd): cache/core_B (60 islands; the core of kb09 and of the scan),
#   core_d02, core_d04 (49 grids each) and core_c11 (47 grids), 12 islands each. Eight W islands of group za start
#   from the best scan chains instead of the plain start set.
cd /home/pradr/2.156/cp_2 || exit 1
BIN=/home/pradr/2.156/cp_2/bin/anneal6
n=0; H=${HOT:-9}
emit() {  # emit <count> <group letters> <core name> <rule> <stage spec> [init file]; @K = size of the core
    local i k spec
    k=$(cat "cache/core_$3.k") || exit 1
    spec=${5//@K/$k}
    for ((i = 0; i < $1; i++)); do
        n=$((n + 1))
        printf 'env ANNEAL_BIN=%s python -u src/islands.py %s%03d %s %s %d %s\n' "$BIN" "$2" "$n" "${6:-cache/core_$3.npy}" "$4" "$((4000 + n))" "$spec"
    done
}
PF="--frozen @K --special @K --boost 1 --dmin 39 --t0 0.45 --t1 0.45 --focus 2 --lam 0.25"   # cold polish, core fixed
PX="--special @K --boost 1 --dmin 39 --t0 0.40 --t1 0.40 --focus 2 --lam 0.2"                # cold polish, core free
hot() { echo "--frozen @K --special @K --boost ${4:-1} --dmin 39 --focus 5 --t0 $1 --t1 $2 --lam $3"; }  # T from $1 to $2
END="own:1x10000000 $PF :: own:1x10000000 $PX"
W() { echo "${H}x20000000 $(hot $1 $1 $2 $3) :: 1x40000000 $(hot $1 0.8 $2 $3) :: $END"; }          # use with rule best
O() { echo "${H}x20000000 $(hot $1 $1 $2 $3) :: 1x40000000 $(hot $1 0.8 $2 $3) :: $END"; }          # use with rule own
C() { echo "1x$((H * 20000000)) $(hot $1 $2 $3) :: 1x40000000 $(hot $2 0.7 $3) :: $END"; }               # use with rule own
{
    # group za: core_B, 60 islands
    emit 6 za B best "$(W 1.75 0.3)"; emit 6 za B best "$(W 2.0 0.6)"; emit 4 za B best "$(W 1.5 0.3)"
    emit 2 za B best "$(W 2.0 0.3)"; emit 2 za B best "$(W 2.0 0.3 2)"; emit 3 za B best "$(W 1.75 0.6)"
    emit 2 za B best "$(W 2.25 0.6)"; emit 2 za B best "$(W 2.25 1.0)"; emit 1 za B best "$(W 1.3 0.3)"
    emit 2 za B best "$(W 2.0 0.6)" ckpt/kt23/state.bin; emit 2 za B best "$(W 1.75 0.3)" ckpt/kt06/state.bin
    emit 2 za B best "$(W 1.75 0.3)" ckpt/kt05/state.bin; emit 2 za B best "$(W 1.5 0.3)" ckpt/kt04/state.bin
    emit 4 za B own "$(O 1.75 0.3)"; emit 4 za B own "$(O 2.0 0.6)"; emit 2 za B own "$(O 1.5 0.3)"; emit 2 za B own "$(O 2.0 0.3)"
    emit 3 za B own "$(C 2.0 1.0 0.3)"; emit 3 za B own "$(C 2.0 1.0 0.6)"; emit 3 za B own "$(C 1.75 1.0 0.3)"; emit 3 za B own "$(C 2.25 1.2 0.6)"
    # groups zb, zc, zd: the other cores, 12 islands each
    for gc in "zb d02" "zc d04" "zd c11"; do
        set -- $gc
        emit 3 $1 $2 best "$(W 1.75 0.3)"; emit 3 $1 $2 best "$(W 2.0 0.6)"; emit 1 $1 $2 best "$(W 1.5 0.3)"; emit 1 $1 $2 best "$(W 2.0 0.3)"
        emit 1 $1 $2 own "$(O 1.75 0.3)"; emit 1 $1 $2 own "$(O 2.0 0.6)"
        emit 1 $1 $2 own "$(C 2.0 1.0 0.3)"; emit 1 $1 $2 own "$(C 2.0 1.0 0.6)"
    done
} > jobs/opt_r5.txt
echo "wrote jobs/opt_r5.txt: $(wc -l < jobs/opt_r5.txt) lines, $(awk '{print $6}' jobs/opt_r5.txt | sort -u | wc -l) distinct tags"
awk '{ s = 0; for (i = 1; i <= NF; i++) if ($i ~ /^([a-z]+:)?[0-9]+x[0-9]+$/) { sub(/^[a-z]+:/, "", $i); split($i, a, "x"); s += a[1] * a[2] } print substr($6, 1, 2), $8, s / 1e6 "M" }' jobs/opt_r5.txt | sort | uniq -c
for f in $(awk '{print $7}' jobs/opt_r5.txt | sort -u); do [ -s "$f" ] || echo "MISSING start set $f"; done
