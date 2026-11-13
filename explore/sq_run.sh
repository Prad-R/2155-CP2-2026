#!/bin/bash
# Squeeze tests (Nov 12): 12 chains of 20M steps from S21 (grids sorted so that the first M are the "special" ones).
#   sq01-sq06: --special M --boost B  (pair penalties of the first M grids multiplied by 1+B), bin/anneal5
#   sq07-sq12: uniform strong pair penalty (--lam 1..4)
# Run inside a job allocation: WBNAME=<job name> explore/wb.sh -c 12 bash explore/sq_run.sh
cd /home/pradr/2.156/cp_2 || exit 1
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal5
COMMON="--init cache/sq_start_S21_sorted.npy --steps 20000000 --dmin 39 --log-every 2000000 --ckpt-every 2000000"
run() { tag=$1; shift; python -u src/anneal.py --tag "$tag" $COMMON "$@" >> "logs/sq-$tag.out" 2>&1 & }
run sq01 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --special 16 --boost 1 --seed 4101
run sq02 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --special 16 --boost 2 --seed 4102
run sq03 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --special 16 --boost 5 --seed 4103
run sq04 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --special 8 --boost 2 --seed 4104
run sq05 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --special 32 --boost 2 --seed 4105
run sq06 --t0 0.45 --t1 0.45 --focus 2 --lam 0.25 --special 16 --boost 2 --seed 4106
run sq07 --t0 0.6 --t1 0.6 --focus 5 --lam 1 --seed 4107
run sq08 --t0 0.6 --t1 0.6 --focus 5 --lam 2 --seed 4108
run sq09 --t0 1.2 --t1 1.2 --focus 5 --lam 2 --seed 4109
run sq10 --t0 1.2 --t1 1.2 --focus 5 --lam 4 --seed 4110
run sq11 --t0 0.45 --t1 0.45 --focus 2 --lam 1 --seed 4111
run sq12 --t0 0.8 --t1 0.8 --focus 5 --lam 1 --lam2 0.1 --dmin2 41 --seed 4112
wait
echo "all squeeze chains finished at $(date)"
