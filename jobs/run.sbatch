#!/bin/bash
# Generic preempt-safe wrapper. Usage (never pi_faez):
#   sbatch -J cp2-name -p mit_normal -t 2:00:00 -c 16 --mem=32G jobs/run.sbatch python -u src/x.py ...
#   sbatch -J cp2-name -p mit_normal_gpu --gres=gpu:1 -t 2:00:00 -c 8 --mem=32G jobs/run.sbatch python -u src/y.py ...
# The python scripts are resumable (per-unit output files, skipped if present), so a
# requeue after preemption/timeout only redoes the unit that was in progress.
#SBATCH -A mit_general
#SBATCH -p mit_normal
#SBATCH -t 1:00:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/%x-%j.out

source /etc/profile.d/modules.sh
module load miniforge/26.7.2-0
conda activate 2.156_cp_2
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$PYTHONPATH"

echo "=== $(date) job $SLURM_JOB_ID on $(hostname) restart=${SLURM_RESTART_COUNT:-0}: $*"
trap 'echo "=== $(date) USR1 received: stopping and requeueing"; kill -TERM "$PID" 2>/dev/null; wait "$PID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
"$@" &
PID=$!
wait "$PID"
rc=$?
echo "=== $(date) finished rc=$rc"
exit $rc
