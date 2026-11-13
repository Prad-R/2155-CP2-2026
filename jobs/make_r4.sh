#!/bin/bash
# Writes jobs/opt_r4.txt: 96 island chains (src/islands.py), tags y001..y096, 300M steps each (~10.5 h).
# Usage: bash jobs/make_r4.sh      Line n of the list is island y<n> (log logs/cp2-optR4-<jobid>-<n>.out).
# Design (from cp2-optR3, see UPDATES.md 2026-11-12 12:45):
#   y001-y048  polish: cold chains started from the best sets of cp2-optR3 (v073 37996, v069 37995, six at 37994),
#              6 stages x 50M steps; rule "best" = each new stage restarts from the best set of the whole run,
#              rule "own" = from the island's own best (keeps the other start sets alive).
#   y049-y084  new sets: 150M warm steps with the lam2 penalty (the setting that produced v073/v069), then
#              3 x 50M cold polish of the island's own record.
#   y085-y096  mild thermal cycling (T 0.60 -> 0.35, five cycles per 50M-step stage) from the best sets.
cd /home/pradr/2.156/cp_2 || exit 1
A="--t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --dmin 39"
B="--t0 0.35 --t1 0.35 --focus 2 --lam 0.15 --dmin 39"
C="--t0 0.40 --t1 0.40 --focus 2 --lam 0.2 --dmin 39"
D="--t0 0.50 --t1 0.50 --focus 2 --lam 0.3 --dmin 39"
E="--t0 0.45 --t1 0.45 --focus 5 --lam 0.25 --dmin 39 --lam2 0.05 --dmin2 41"
Y="--t0 0.60 --t1 0.35 --cycles 5 --focus 2 --lam 0.3 --dmin 39"
P=6x50000000
n=0
emit() {  # emit <count> <init tag> <rule> <stage spec>
    local i
    for ((i = 0; i < $1; i++)); do
        n=$((n + 1))
        printf 'python -u src/islands.py y%03d ckpt/%s/state.bin %s %d %s\n' "$n" "$2" "$3" "$((3000 + n))" "$4"
    done
}
{
    # polish islands (48)
    emit 1 v073 own "$P $A"; emit 5 v073 best "$P $A"; emit 1 v073 own "$P $C"; emit 2 v073 best "$P $C"
    emit 1 v073 own "$P $B"; emit 2 v073 best "$P $B"; emit 2 v073 best "$P $D"; emit 1 v073 own "$P $E"; emit 1 v073 best "$P $E"
    emit 1 v069 own "$P $A"; emit 3 v069 best "$P $A"; emit 1 v069 own "$P $C"; emit 1 v069 best "$P $C"
    emit 1 v069 own "$P $B"; emit 1 v069 best "$P $B"; emit 1 v069 best "$P $D"; emit 1 v069 best "$P $E"
    for s in v018 v070 v072 v074; do
        emit 1 $s own "$P $A"; emit 1 $s best "$P $A"; emit 1 $s best "$P $C"; emit 1 $s own "$P $B"; emit 1 $s best "$P $D"
    done
    emit 1 v090 own "$P $C"; emit 1 v092 own "$P $A"
    # new sets: warm lam2 stage, then cold polish of the island's own record (36)
    for T in 0.70 0.75 0.80; do
        W="1x150000000 --t0 $T --t1 $T --focus 5 --lam 0.3 --dmin 39 --lam2 0.1 --dmin2 41 :: 3x50000000 $A"
        for k in 1 2 3 4 5 6; do emit 1 r761 own "$W"; emit 1 r752 own "$W"; done
    done
    # mild thermal cycling (12)
    emit 6 v073 best "$P $Y"; emit 3 v069 best "$P $Y"
    for s in v070 v072 v074; do emit 1 $s own "$P $Y"; done
} > jobs/opt_r4.txt
echo "wrote jobs/opt_r4.txt: $(wc -l < jobs/opt_r4.txt) lines, $(awk '{print $4}' jobs/opt_r4.txt | sort -u | wc -l) distinct tags"
