#!/bin/bash
# quick checks of bin/anneal5 on a compute node: rules self-test, start-set prep, 200k-step runs with and without --special
cd /home/pradr/2.156/cp_2 || exit 1
export ANNEAL_BIN=/home/pradr/2.156/cp_2/bin/anneal5
python -u src/anneal.py selftest 2>&1 | tail -n 2
python explore/sq_prep.py submissions/S21_20261112_154245.npy cache/sq_start_S21_sorted.npy
rm -rf ckpt/sqt1 ckpt/sqt2
python -u src/anneal.py --tag sqt1 --init cache/sq_start_S21_sorted.npy --steps 200000 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --dmin 39 --seed 1 --log-every 100000 --ckpt-every 100000 2>&1 | grep -v "^exec" | cut -c1-200 &
python -u src/anneal.py --tag sqt2 --init cache/sq_start_S21_sorted.npy --steps 200000 --t0 0.6 --t1 0.6 --focus 5 --lam 0.3 --dmin 39 --special 16 --boost 5 --seed 1 --log-every 50000 --ckpt-every 100000 2>&1 | grep -v "^exec" | cut -c1-200 &
wait
rm -rf ckpt/sqt1 ckpt/sqt2
