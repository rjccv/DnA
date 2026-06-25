#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

torchrun --nproc_per_node=1 main.py \
  --model dna_base_patch16_224_shared_values \
  --batch-size 256 \
  --grad_accum_steps 4 \
  --input-size 224 \
  --data-path /datasets/ImageNet2012nonpub \
  --data-set IMNET \
  --no-model-ema \
  --epochs 300 \
  --num_workers 8 \
  --weight-decay 0.075 \
  --drop-path 0.15 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --output_dir ./runs2/dna_shared_values \
  --run_name dna_shared_values \
  --wandb_entity personal_exps \
  --wandb_project uncategorized \
  "$@"
