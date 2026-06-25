#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32768
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --time=80:00:00
#SBATCH --error=outs/%J.out
#SBATCH --output=outs/%J.out
#SBATCH --job-name=cv-nntu

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

srun --label --nodes=1 --ntasks=1 --gpus-per-task=2 \
    python tools/run_net.py \
    --cfg configs/NTU/cv_dna.yaml \
    NUM_GPUS 2 DATA_LOADER.NUM_WORKERS 8

srun --label --nodes=1 --ntasks=1 --gpus-per-task=2 \
    python tools/run_net.py \
    --cfg configs/NTU/cs_dna.yaml \
    NUM_GPUS 2 DATA_LOADER.NUM_WORKERS 8


