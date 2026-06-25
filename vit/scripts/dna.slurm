#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=180:00:00
#SBATCH --error=outs/%J.out
#SBATCH --output=outs/%J.out
#SBATCH --job-name=vit_dna

module purge
module load cuda/cuda-12.1
source activate dna
module list

## Run job ##
cd "$(dirname "$0")/.."


srun --exclusive --nodes=1 --ntasks=1 torchrun --nproc_per_node=1 --master_port 20200 main.py --model dna_vit_base_patch16_224 --batch-size 256 --grad_accum_steps 4 --input-size 224 --data-path /datasets/ImageNet2012nonpub --data-set IMNET --no-model-ema --epochs 300 --num_workers 8 --weight-decay 0.075 --drop-path 0.15 --mixup 0.8 --cutmix 1.0 --output_dir ./runs/dna_ckpts --run_name dna
