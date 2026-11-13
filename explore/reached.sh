#!/bin/bash
# Exit 0 if some C++ chain already holds an all-valid set scoring >= $1 (default 38000), else exit 1.
# Reads only the 8-byte "best" field of each ckpt/<tag>/state.bin header, so it needs no Python.
cd /home/pradr/2.156/cp_2 || exit 2
target=${1:-38000}
best=0; tag=none
for f in ckpt/*/state.bin; do
    b=$(od -A n -t d8 -j 24 -N 8 "$f" 2>/dev/null | tr -d ' ')
    if [ -n "$b" ] && [ "$b" -gt "$best" ]; then best=$b; tag=$(basename "$(dirname "$f")"); fi
done
echo "best set so far: $best (chain $tag); target $target"
[ "$best" -ge "$target" ]
