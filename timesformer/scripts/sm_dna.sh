#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32768
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=14:00:00
#SBATCH --error=outs/%J.out
#SBATCH --output=outs/%J.out
#SBATCH --job-name=sm-a

# --- Environment Setup ---
module purge
module load cuda/cuda-12.1
source activate timesformer

WORKINGDIR=/path/to/TimeSformer
export PYTHONPATH="./:$PYTHONPATH"

cd $WORKINGDIR

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print("device_count:", torch.cuda.device_count())
PY

srun --label --nodes=1 --ntasks=1 \
    python tools/run_net.py \
    --cfg configs/Smarthome/dna_cross_view_2.yaml \
    NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 8

srun --label --nodes=1 --ntasks=1 \
    python tools/run_net.py \
    --cfg configs/Smarthome/dna_cross_subject.yaml \
    NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 8
