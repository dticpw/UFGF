#!/usr/bin/env bash
# Drop baseline (2026-11 revision): same data/noise/lr schedule and the same in-batch clean mask as UFF,
# but flagged pairs are removed from the batch instead of being filled with the UFR (UFGF_MODE=drop in main_uff.py).
# Set UFGF_ORACLE=1 to drop with the ground-truth noise labels. The UFR argument is only loaded, never used.
# Usage: bash revision/run_drop.sh <dataset> <dataset_root> <noise_ratio> <seed> <run_name> <any_ufr.pt> [epochs]
set -euo pipefail
DATASET=$1; ROOT=$2; NOISE=$3; SEED=$4; NAME=$5; UFR=$6; EPOCHS=${7:-5}
PY=${UFGF_PY:-/root/miniconda3/envs/t200/bin/python}
RUN=${UFGF_RUNS:-/root/autodl-tmp/runs}/$NAME
mkdir -p "$RUN/drop"
export PYTHONPATH="$PWD/revision:$PWD"
echo "[$(date)] drop start" | tee -a "$RUN/run.log"
UFGF_MODE=drop UFGF_RUN_DIR=$RUN/drop UFGF_UFR=$UFR "$PY" -u revision/main_uff.py \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --noise_ratio $NOISE \
  --batch_size ${UFGF_BS:-256} --epochs $EPOCHS --lr 1e-7 --seed $SEED >> "$RUN/drop/stdout.log" 2>&1
BEST=$(grep -o 'The best model is: [^,]*' "$RUN/drop/log.txt" | tail -1 | sed 's/The best model is: //')
echo "[$(date)] test drop best=$BEST" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/drop/test "$PY" -u revision/main_ufg.py --eval --resume "$BEST" \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --eval_batch_size 64 \
  >> "$RUN/drop/test_stdout.log" 2>&1
echo "[$(date)] done" | tee -a "$RUN/run.log"
