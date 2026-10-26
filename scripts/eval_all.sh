#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to the base-model directory or Hub identifier}"
: "${CHECKPOINT:?Set CHECKPOINT to a PMOPD snapshot directory, such as OUTPUT_DIR/final}"
: "${EVAL_DATA_DIR:?Set EVAL_DATA_DIR to a directory with math.jsonl, reason.jsonl, and code.jsonl}"
: "${EVAL_OUTPUT_DIR:?Set EVAL_OUTPUT_DIR for prediction and summary files}"

mkdir -p "$EVAL_OUTPUT_DIR"
GPU="${GPU:-0}"
python evaluation/eval_math.py \
  --model "$BASE_MODEL" --override "$CHECKPOINT" --data "$EVAL_DATA_DIR/math.jsonl" \
  --output "$EVAL_OUTPUT_DIR/math.jsonl" --gpu "$GPU" --batch-size "${MATH_BATCH_SIZE:-8}"
python evaluation/eval_reason.py \
  --model "$BASE_MODEL" --override "$CHECKPOINT" --data "$EVAL_DATA_DIR/reason.jsonl" \
  --output "$EVAL_OUTPUT_DIR/reason.jsonl" --gpu "$GPU" --batch-size "${REASON_BATCH_SIZE:-8}"
python evaluation/eval_code.py \
  --model "$BASE_MODEL" --override "$CHECKPOINT" --data "$EVAL_DATA_DIR/code.jsonl" \
  --output "$EVAL_OUTPUT_DIR/code.jsonl" --gpu "$GPU" --batch-size "${CODE_BATCH_SIZE:-4}" \
  --code-timeout "${CODE_TIMEOUT:-10}"
