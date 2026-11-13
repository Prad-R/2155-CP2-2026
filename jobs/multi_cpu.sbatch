#!/bin/bash
# Run several resumable commands concurrently on CPUs only (one thread each by default), requeue-safe.
# Usage: sbatch -J cp2-name -c 8 --mem=24G -t 3:00:00 jobs/multi_cpu.sbatch cmds.txt   (never pi_faez)
#   cmds.txt: one shell command per line; each gets its own log logs/<jobname>-<jobid>-<n>.out
#SBATCH -A mit_general
#SBATCH -p mit_preemptable
#SBATCH -t 2:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/%x-%j.out

source /etc/profile.d/modules.sh
module load miniforge/26.7.2-0
conda activate 2.156_cp_2
cd "$SLURM_SUBMIT_DIR"
# Optional guard: export CP2_SKIP_IF="<command>" before sbatch. If the command succeeds at start (or after a
# requeue), the job is not needed any more and exits at once without running anything.
if [ -n "${CP2_SKIP_IF:-}" ] && bash -c "$CP2_SKIP_IF"; then
    echo "=== $(date) job $SLURM_JOB_ID: skip condition met ($CP2_SKIP_IF), nothing to run"; exit 0
fi
NCMD=$(grep -cv '^\s*$' "$1")
T=$(( ${SLURM_CPUS_PER_TASK:-4} / NCMD )); [ "$T" -lt 1 ] && T=1
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$PYTHONPATH"
echo "=== $(date) job $SLURM_JOB_ID on $(hostname) restart=${SLURM_RESTART_COUNT:-0}: $NCMD commands from $1, $T threads each"
PIDS=()
n=0
while IFS= read -r line; do
    [ -z "${line// }" ] && continue
    n=$((n+1))
    bash -c "exec $line" >> "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-$n.out" 2>&1 &
    PIDS+=($!)
done < "$1"
trap 'echo "=== $(date) USR1: stopping and requeueing"; kill -TERM "${PIDS[@]}" 2>/dev/null; wait; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
echo "=== $(date) finished rc=$rc"
exit $rc
