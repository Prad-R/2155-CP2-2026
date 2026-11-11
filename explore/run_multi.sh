#!/bin/bash
# run each line of $1 as a background process (one thread each), log to $2-<n>.out, wait for all
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
n=0
while IFS= read -r line; do
  [ -z "${line// }" ] && continue
  n=$((n+1))
  bash -c "exec $line" >> "$2-$n.out" 2>&1 &
done < "$1"
wait
