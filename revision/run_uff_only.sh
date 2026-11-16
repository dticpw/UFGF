#!/usr/bin/env bash
# UFF stage only, with a given fixed anchor (ablation: anchor not produced by UFG), then test.
# Usage: bash revision/run_uff_only.sh <dataset> <dataset_root> <noise_ratio> <seed> <run_name> <ufr.pt>
set -euo pipefail
DATASET=$1; ROOT=$2; NOISE=$3; SEED=$4; NAME=$5; UFR=$6
PY=${UFGF_PY:-/root/miniconda3/envs/t200/bin/python}
RUN=${UFGF_RUNS:-/root/autodl-tmp/runs}/$NAME
mkdir -p "$RUN/uff"
export PYTHONPATH="$PWD/revision:$PWD"
echo "[$(date)] UFF-only start with $UFR" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/uff UFGF_UFR=$UFR "$PY" -u revision/main_uff.py \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --noise_ratio $NOISE \
  --batch_size ${UFGF_BS:-256} --epochs 5 --lr 1e-7 --seed $SEED >> "$RUN/uff/stdout.log" 2>&1
BEST=$(grep -o 'The best model is: [^,]*' "$RUN/uff/log.txt" | tail -1 | sed 's/The best model is: //')
echo "[$(date)] test uff best=$BEST" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/uff/test "$PY" -u revision/main_ufg.py --eval --resume "$BEST" \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --eval_batch_size 64 \
  >> "$RUN/uff/test_stdout.log" 2>&1
echo "[$(date)] done" | tee -a "$RUN/run.log"
