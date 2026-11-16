#!/usr/bin/env bash
# Two-stage UFGF run used for the 2026-11 revision (UFG -> fixed UFR -> UFF -> test).
# Usage: bash revision/run_two_stage.sh <dataset f30k|coco> <dataset_root> <noise_ratio> <seed> <run_name>
# Optional env: UFGF_BS (batch size, default 256), UFGF_SIGMA (UFG replacement threshold, default 0.2),
#   UFGF_NANNS (captions per image: 5 for f30k/coco, 1 for cc; default 5), UFGF_LAMBDA1 (default 4; 0 removes L_I/L_T),
#   UFGF_UFF_RULE=three (UFF uses the UFG three-condition replacement rule instead of all non-clean).
# Run from the code directory (the one containing main_NPC.py and revision/).
set -euo pipefail
DATASET=$1; ROOT=$2; NOISE=$3; SEED=$4; NAME=$5
PY=${UFGF_PY:-/root/miniconda3/envs/t200/bin/python}
RUNS=${UFGF_RUNS:-/root/autodl-tmp/runs}
RUN=$RUNS/$NAME
mkdir -p "$RUN/ufg" "$RUN/uff"
COMMON="--dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --noise_ratio $NOISE --batch_size ${UFGF_BS:-256} --epochs 5 --lr 1e-7 --seed $SEED"
export PYTHONPATH="$PWD/revision:$PWD"

echo "[$(date)] UFG start" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/ufg "$PY" -u revision/main_ufg.py $COMMON --lambda1 ${UFGF_LAMBDA1:-4} --lambda2 0.9 >> "$RUN/ufg/stdout.log" 2>&1

UFR=$RUN/ufg/ufr/universal_pseudo_text3.pt   # same extraction point as the archived pipeline (epoch 4, first batch)
test -f "$UFR"
echo "[$(date)] UFF start with $UFR" | tee -a "$RUN/run.log"
UFGF_RUN_DIR=$RUN/uff UFGF_UFR=$UFR "$PY" -u revision/main_uff.py $COMMON >> "$RUN/uff/stdout.log" 2>&1

for STAGE in ufg uff; do
  BEST=$(grep -o 'The best model is: [^,]*' "$RUN/$STAGE/log.txt" | tail -1 | sed 's/The best model is: //')
  echo "[$(date)] test $STAGE best=$BEST" | tee -a "$RUN/run.log"
  UFGF_RUN_DIR=$RUN/$STAGE/test "$PY" -u revision/main_ufg.py --eval --resume "$BEST" \
    --dataset $DATASET --dataset_root $ROOT --vision_model ViT-B/32 --num_anns ${UFGF_NANNS:-5} --eval_batch_size 64 \
    >> "$RUN/$STAGE/test_stdout.log" 2>&1
done
echo "[$(date)] done" | tee -a "$RUN/run.log"
