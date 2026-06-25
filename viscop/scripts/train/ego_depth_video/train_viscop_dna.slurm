#!/bin/bash
#SBATCH --job-name=train_viscop_dna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00
#SBATCH --output=outs/%x-%j.out
#SBATCH --error=outs/%x-%j.out

# Slurm Environment Config
export MASTER_ADDR_PASSED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ARG_WORLD_SIZE=${SLURM_NNODES:-1}
ARG_NPROC_PER_NODE=2 

if [[ -v MASTER_ADDR_PASSED ]]; then
    ARG_MASTER_ADDR=$MASTER_ADDR_PASSED 
else
    ARG_MASTER_ADDR=127.0.0.1 
fi
ARG_MASTER_PORT=12999
ARG_RANK=${SLURM_NODEID:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi

if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "MASTER_ADDR: $MASTER_ADDR. MASTER_PORT: $MASTER_PORT. RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128 # aka effective batch size
LOCAL_BATCH_SIZE=4 # batch size per GPU
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

INIT_MODEL=/path/to/weights/videollama3_7b_local/ # path to base VLM (for ViSCoP we use VideoLLaMA3 as the base VLM)

NUM_DATA_WORKERS=8
NUM_TRAIN_EPOCHS=3
LORA_TRAINING=True

# ViSCoP Arguments
NUM_VISUAL_PROBES=16
INTERACTION_MODULE_POS=all
ATTENTION_TYPE=cross_denoising_attention
PASS_PROBES_TO_LLM=True
PASS_VIS_FEATURES_TO_LLM=True

# Logging Arguments
export WANDB_PROJECT=egoexo
REPORT_TO=wandb
OUTP_DIR=work_dirs/egoexo
RUN_NAME=ADeIT_qwen2.5_7b_EgoExo4D-CAv1-LLM_LoRA

DATASET_CACHE_DIR=/path/to/.cache/huggingface/datasets/viscop_datasetcache

# Data Arguments
DATA_DIR=/path/to/DnA/viscop/data/egocentric_videosz
TRAINING_JSON="/path/to/DnA/viscop/data/egoexo4d-vllama3-S4cap-egoview.json"

if [[ $TRAINING_JSON == *"egoview"* ]]; then
    MAX_FRAMES=40 # use 40 frames for training on ego
else
    MAX_FRAMES=180
fi

# Optional Arguments. Set TESTING to 1 to quickly test the training script without logging or data workers, useful for debugging
TESTING=0
if [ $TESTING -eq 1 ]; then
    NUM_DATA_WORKERS=0
    REPORT_TO=none
    RUN_NAME=TESTING
fi

mkdir -p "${OUTP_DIR}/${RUN_NAME}/"
cp "$0" "${OUTP_DIR}/${RUN_NAME}/"

module purge
module load ffmpeg/ffmpeg-7.0.2-gcc-12.2.0
module load cuda/cuda-12.4.0
module load gcc/gcc-12.2.0
source activate viscop

cd /path/to/DnA/viscop

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    viscop/train_viscop.py \
    --interaction_module_layers $INTERACTION_MODULE_POS \
    --interaction_module $ATTENTION_TYPE \
    --lora_enable $LORA_TRAINING \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --deepspeed scripts/zero1.json \
    --model_type viscop_qwen2 \
    --model_path $INIT_MODEL \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path $TRAINING_JSON \
    --data_folder $DATA_DIR \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames $MAX_FRAMES \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${RUN_NAME} \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --mm_projector_lr 1e-5 \
    --llm_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers $NUM_DATA_WORKERS \
    --report_to $REPORT_TO \
    --run_name $RUN_NAME \
    --dataset_cache_dir $DATASET_CACHE_DIR \
    --include_visual_tokens $PASS_VIS_FEATURES_TO_LLM \
    --include_visual_probes $PASS_PROBES_TO_LLM \
    --num_visual_probes $NUM_VISUAL_PROBES \
    --warm_init_interaction_module True \
    --warm_init_value_im 1e-4