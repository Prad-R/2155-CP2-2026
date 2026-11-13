#!/bin/bash
# Quick end-to-end check of one line of jobs/opt_r5.txt with the steps scaled down (about 2 minutes, 1 core):
#   WBNAME=<job name> explore/wb.sh -c 2 bash explore/test_r5.sh [line number]
# Checks that the island runs its hot stage and both polish stages, that every stage is verified by the Python
# rules, and that the core rows are unchanged wherever --frozen was given.
cd /home/pradr/2.156/cp_2 || exit 1
ln=${1:-1}
line=$(sed -n "${ln}p" jobs/opt_r5.txt | sed -E 's/ z[a-d][0-9]{3} / zq01 /; s/9x20000000/3x150000/g; s/1x180000000/1x300000/g; s/1x40000000/1x150000/g; s/1x10000000/1x100000/g')
init=$(echo "$line" | awk '{print $7}')
rm -rf ckpt/zq01s[0-9]*
echo "test line: $(echo "$line" | cut -c1-160) ..."
bash -c "exec $line" 2>&1 | grep -E "^===|^final|^finished|failed|rror" | cut -c1-200
python -c "
import glob, sys, numpy as np
sys.path.insert(0, 'src')
from anneal import read_state
init = '$init'; k = int(open(init[:-4] + '.k').read()); a = np.load(init)
for d in sorted(glob.glob('ckpt/zq01s[0-9]*')):
    st = read_state(d + '/state.bin')
    print(d, 'best', int(st['best']) if 'best' in st else '?', '| core rows unchanged:', bool((st['bestG'][:k] == a[:k]).all()))
"
rm -rf ckpt/zq01s[0-9]*
