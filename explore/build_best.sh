#!/bin/bash
# Build a submission file from the best verified exact-validity set found so far (chains may still be running).
# Usage: explore/build_best.sh S19 ["purpose text"]
#        explore/build_best.sh next ["purpose text"]   (next free S number; does nothing unless a chain beats every built submission)
# Runs on a compute node through explore/wb.sh (inside the workbench job if one is running, otherwise a short srun).
name=${1:?usage: explore/build_best.sh <name, e.g. S19> [purpose]}
purpose=${2:-best exact-validity set so far}
cd /home/pradr/2.156/cp_2 || exit 1
if [ "$name" = next ]; then
    # "next": build only if some chain beats every submission built so far, and give it the next free S number.
    have=$(grep -o '"expected_int": [0-9]*' results/submission_log.jsonl | awk '{if ($2 > m) m = $2} END {print m + 0}')
    last=$(grep -o '"name": "S[0-9]*"' results/submission_log.jsonl | grep -o '[0-9]*' | sort -n | tail -n 1)
    if ! explore/reached.sh $((have + 1)); then echo "no set better than the best submission built so far ($have): nothing to build"; exit 0; fi
    name=S$((last + 1))
fi
if [ -z "${CP2_INSIDE:-}" ]; then exec explore/wb.sh -c 2 env CP2_INSIDE=1 bash explore/build_best.sh "$name" "$purpose"; fi
list=$(python -u src/best_chains.py --top 3 --verify) || { echo "$list"; exit 1; }
echo "$list"
tag=$(echo "$list" | awk '$NF=="verified=YES"{print $1; exit}')
if [ -z "$tag" ]; then echo "no verified chain found"; exit 1; fi
python -u src/make_exact_submission.py --name "$name" --source "ckpt/$tag/state.bin" --purpose "$purpose (chain $tag)"
