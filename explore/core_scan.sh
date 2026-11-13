#!/bin/bash
# Temperature scan with a fixed core (Nov 12, 23:30). The cooling chains of explore/core_cool.sh reached 38015 from a
# start at T 1.5 and only 37947-37968 from a start at T 0.9-1.0, so: what score does a chain hold at constant T
# between 1.3 and 4, and does the fixed core matter at that temperature? 26 constant-T chains, logged every 500k
# steps; they run until they are stopped (end of the job).
#   WBNAME=<job name> explore/wb.sh -c 96 bash explore/core_scan.sh <start.npy> <tag prefix> <seed base> <steps>
cd /home/pradr/2.156/cp_2 || exit 1
start=${1:?start set}; pre=${2:?tag prefix}; sb=${3:?seed base}; steps=${4:?steps}
K=$(cat "${start%.npy}.k") || exit 1
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal6
COMMON="--init $start --steps $steps --dmin 39 --log-every 500000 --ckpt-every 1000000 --special $K --boost 1"
run() { n=$1; t=$2; shift 2; python -u src/anneal.py --tag "$pre$n" $COMMON --seed $((sb + 10#$n)) --t0 "$t" --t1 "$t" "$@" >> "logs/core-$pre$n.out" 2>&1 & }
F="--frozen $K"
run 01 1.3  $F --focus 5 --lam 0.3
run 02 1.3  $F --focus 5 --lam 0.3
run 03 1.5  $F --focus 5 --lam 0.3
run 04 1.5  $F --focus 5 --lam 0.3
run 05 1.75 $F --focus 5 --lam 0.3
run 06 1.75 $F --focus 5 --lam 0.3
run 07 2.0  $F --focus 5 --lam 0.3
run 08 2.0  $F --focus 5 --lam 0.3
run 09 2.5  $F --focus 5 --lam 0.3
run 10 2.5  $F --focus 5 --lam 0.3
run 11 3.0  $F --focus 5 --lam 0.3
run 12 3.0  $F --focus 5 --lam 0.3
run 13 4.0  $F --focus 5 --lam 0.3
run 14 4.0  $F --focus 5 --lam 0.3
run 15 1.5  --focus 5 --lam 0.3
run 16 1.5  --focus 5 --lam 0.3
run 17 1.5  $F --focus 2 --lam 0.3
run 18 1.5  $F --focus 2 --lam 0.3
run 19 1.5  $F --focus 5
run 20 1.5  $F --focus 5
run 21 1.5  $F --focus 5 --lam 0.6
run 22 1.5  $F --focus 5 --lam 0.6
run 23 2.0  $F --focus 5 --lam 0.6
run 24 2.0  $F --focus 5 --lam 0.6
run 25 1.5  $F --focus 10 --lam 0.3
run 26 1.5  $F --focus 10 --lam 0.3
wait
echo "scan chains $pre ended at $(date)"
