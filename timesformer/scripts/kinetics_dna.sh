#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --time=100:00:00
#SBATCH --error=outs/%J.out
#SBATCH --output=outs/%J.out
#SBATCH --job-name=k400-a2

module purge
module load cuda/cuda-12.1
source activate timesformer

WORKINGDIR=/path/to/TimeSformer
export PYTHONPATH="./:$PYTHONPATH"

cd $WORKINGDIR

srun --label --nodes=1 --ntasks=1 \
    python tools/run_net.py \
    --cfg configs/Kinetics/dna.yaml \
    NUM_GPUS 2 DATA_LOADER.NUM_WORKERS 8

