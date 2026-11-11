#!/bin/bash
# Create/refresh the conda env for CP2. Run on a compute node, e.g.:
#   srun -A mit_general -p mit_normal -t 0:30:00 -c 4 --mem=8G bash scripts/setup_env.sh
# Idempotent: re-running only installs what is missing.
set -eo pipefail

ENV_NAME=2.156_cp_2
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge/26.7.2-0

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -q -n "$ENV_NAME" python=3.10 pip
fi
conda activate "$ENV_NAME"

pip install -q --no-cache-dir -r "$REPO_DIR/requirements.txt"
pip install -q --no-cache-dir torch lightgbm xgboost

python - <<'EOF'
import sys, numpy, scipy, sklearn, pandas, matplotlib, torch, lightgbm, xgboost
print("python", sys.version.split()[0])
for m in (numpy, scipy, sklearn, pandas, matplotlib, torch, lightgbm, xgboost):
    print(m.__name__, m.__version__)
print("torch cuda build:", torch.version.cuda, "| cuda available here:", torch.cuda.is_available())
EOF
echo "ENV_OK $(hostname)"
