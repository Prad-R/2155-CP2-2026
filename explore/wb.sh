#!/bin/bash
# Run a command inside the workbench job (cp2-wb) on a compute node.
# Usage: explore/wb.sh [-c NCPU] python -u explore/x.py ...
NC=8
if [ "$1" = "-c" ]; then NC=$2; shift 2; fi
JID=$(squeue -u "$USER" -h -n "${WBNAME:-cp2-wb}" -t RUNNING -o %i | head -1)
if [ -n "$JID" ]; then
  RUN=(srun --jobid="$JID" --overlap --cpu-bind=none -n 1 -c "$NC")
else
  # No workbench job running: short allocation of its own, for light steps only (never pi_faez).
  [ "$NC" -gt 4 ] && NC=4
  RUN=(srun -A mit_general -p "${WBPART:-mit_preemptable}" -t 0:15:00 -J cp2-light -n 1 -c "$NC" --mem=6G)
fi
cd /home/pradr/2.156/cp_2
"${RUN[@]}" bash -c "
source /etc/profile.d/modules.sh
module load miniforge/26.7.2-0
conda activate 2.156_cp_2
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC
export PYTHONPATH=/home/pradr/2.156/cp_2:/home/pradr/2.156/cp_2/src:/home/pradr/2.156/cp_2/explore:\$PYTHONPATH
$(printf '%q ' "$@")
"
