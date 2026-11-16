#!/usr/bin/env bash
# Plain CLIP fine-tuning baseline under the same data/noise/lr schedule (main_uff.py with UFGF_MODE=vanilla:
# no replacement, mean CE over all pairs). The UFR argument is only loaded, never used.
# Usage: bash revision/run_vanilla.sh <dataset> <dataset_root> <noise_ratio> <seed> <run_name> <any_ufr.pt> [epochs]
set -euo pipefail
DATASET=$1; ROOT=$2; NOISE=$3; SEED=$4; NAME=$5; UFR=$6; EPOCHS=${7:-5}
PY=${UFGF_PY:-/root/miniconda3/envs/t200/bin/python}
RUN=${UFGF_RUNS:-/root/autodl-tmp/runs}/$NAME
mkdir -p "$RUN/vanilla"
export PYTHONPATH="$PWD/revision:$PWD"
echo "[$(date)] vanilla start" | tee -a "$RUN/run.log"
UFGF_MODE=vanilla UFGF_RUN_DIR=$RUN/vanilla UFGF_UFR=$UFR "$PY" -u revision/main_uff.py \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --noise_ratio $NOISE \
  --batch_size ${UFGF_BS:-256} --epochs $EPOCHS --lr 1e-7 --seed $SEED >> "$RUN/vanilla/stdout.log" 2>&1
BEST=$(grep -o 'The best model is: [^,]*' "$RUN/vanilla/log.txt" | tail -1 | sed 's/The best model is: //')
echo "[$(date)] test vanilla best=$BEST" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/vanilla/test "$PY" -u revision/main_ufg.py --eval --resume "$BEST" \
  --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --eval_batch_size 64 \
  >> "$RUN/vanilla/test_stdout.log" 2>&1
echo "[$(date)] done" | tee -a "$RUN/run.log"
