#!/bin/bash
# Build bin/anneal6 and check the new options end to end (about a minute, 2 cores):
#   WBNAME=<job name> explore/wb.sh -c 2 bash explore/core_quick.sh
cd /home/pradr/2.156/cp_2 || exit 1
echo "cores visible to this step: $(nproc)"
g++ -O3 -mavx2 -std=c++17 -o bin/anneal6 src/anneal.cpp || exit 1
export ANNEAL_BIN=$PWD/bin/anneal6
python src/anneal.py selftest || exit 1
python explore/core_prep.py ckpt/crystal/c11.bin cache/core_c11.npy || exit 1
K=$(cat cache/core_c11.k)
rm -rf ckpt/ktest
python -u src/anneal.py --tag ktest --init cache/core_c11.npy --steps 400000 --dmin 39 --log-every 100000 --ckpt-every 100000 \
    --special "$K" --frozen "$K" --t0 0.5 --t1 0.5 --focus 5 --lam 0.3 --boost 1 --seed 1 2>&1 | cut -c1-230
python -c "
import numpy as np, sys
sys.path.insert(0, 'src')
from anneal import read_state
k = int(open('cache/core_c11.k').read()); a = np.load('cache/core_c11.npy'); st = read_state('ckpt/ktest/state.bin')
print('frozen rows unchanged:', bool((st['G'][:k] == a[:k]).all() and (st['bestG'][:k] == a[:k]).all()), '| other rows changed cells per grid:', (st['bestG'][k:] != a[k:]).reshape(100 - k, 49).sum(1).tolist())
"
rm -rf ckpt/ktest
bin/anneal6 --crystal --cinit ckpt/crystal/c11.bin --state ckpt/crystal/qtest.bin --steps 20000 --t0 0.05 --t1 0.02 --keep 60 --cstep 0.2 --log-every 10000 --thr 70 17 -187 -1155 --seed 5 2>&1 | cut -c1-200
rm -f ckpt/crystal/qtest.bin
