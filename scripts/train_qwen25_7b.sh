#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to a Qwen2.5 base-model directory or Hub identifier}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing math.jsonl, code.jsonl, and reason.jsonl}"
: "${TEACHER_MATH:?Set TEACHER_MATH to the math teacher adapter/snapshot}"
: "${TEACHER_CODE:?Set TEACHER_CODE to the code teacher adapter/snapshot}"
: "${TEACHER_REASON:?Set TEACHER_REASON to the reasoning teacher adapter/snapshot}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
python -m pmopd.train \
  --mode pmopd \
  --task_order code reason math \
  --base "$BASE_MODEL" \
  --data_dir "$DATA_DIR" \
  --teacher_math "$TEACHER_MATH" \
  --teacher_code "$TEACHER_CODE" \
  --teacher_reason "$TEACHER_REASON" \
  --output_dir "$OUTPUT_DIR" \
  --gpu 0 --teacher_gpus 1 2 3 \
  --full_data_epoch --epochs 1 --cycles 4 \
  --batch 16 --grad_accum 3 --lr 2e-6 \
  --subspace_rank 16 --kl_direction reverse --temperature 1.0 --top_k_loss 256 \
  --max_prompt 768 --max_prompt_code 2048 --max_new 768 \
  --save_every 0 --sample_every 100 --seed 0 --svd_device cpu \
  --save_all_stages
