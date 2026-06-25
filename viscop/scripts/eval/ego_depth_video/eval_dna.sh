#!/bin/bash
#SBATCH --job-name=dna_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=metrics/%x-%j.out
#SBATCH --error=metrics/%x-%j.out

module purges
module load ffmpeg/ffmpeg-7.0.2-gcc-12.2.0
module load cuda/cuda-12.4.0
module load gcc/gcc-12.2.0
source activate viscop_env


# --- 2. DEFINE ARGS ---
MODEL="/path/to/DnA/viscop/work_dirs/egoexo/DnA_qwen2.5_7b_LLM_LoRA"
BENCHMARKS="egoperceptionmcq,egoschema"
NODES=1

export MASTER_PORT=$((12355 + ($SLURM_JOB_ID % 1000)))
export OLLAMA_PORT=$((15000 + ($SLURM_JOB_ID % 1000)))

GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected $GPUS GPUs for this job."

cd /path/to/DnA/viscop

# --- 3. RUN THE LOGIC SCRIPT ---
# Passes the detected GPU count automatically
bash scripts/eval/ego_depth_video/eval_video_and_metrics.sh "$MODEL" "$BENCHMARKS" $NODES $GPUS
