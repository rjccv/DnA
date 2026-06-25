#!/bin/bash
MODEL_PATH=${1}
BENCHMARKS=${2:-"videomme,egoschema,nextqa,egoperceptionmcq,adlx_mcq,adlx_descriptions"}
ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-1}
DEBUG_MODE=${5:-false}


# --- SLURM/DISTRIBUTED SETUP ---
if [[ -v MASTER_ADDR_PASSED ]]; then
    ARG_MASTER_ADDR=$MASTER_ADDR_PASSED 
else
    ARG_MASTER_ADDR=127.0.0.1 
fi
ARG_MASTER_PORT=${MASTER_PORT:-12355}
ARG_RANK=${SLURM_NODEID:-0} # Safe default

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "BENCHMARKS: $BENCHMARKS"
SAVE_DIR=local_evaluations/egoview/$(basename $MODEL_PATH)
# FIXED: Use the correct absolute path found in previous steps
DATA_ROOT=/path/to/VisCop/vlm_eval_bench
declare -A DATA_ROOTS

mkdir -p "${SAVE_DIR}"
CMD="$(ps -o args= -p $$)"
echo "$CMD" > "$SAVE_DIR/run_command"

# --- DEFINE DATA ROOTS ---
DATA_ROOTS["videomme"]="$DATA_ROOT/videomme"
DATA_ROOTS["egoschema"]="$DATA_ROOT/egoschema"
DATA_ROOTS["nextqa"]="$DATA_ROOT/nextqa"
DATA_ROOTS["egoperceptionmcq"]="$DATA_ROOT/egoperceptionmcq"
DATA_ROOTS["egoperceptionmcq_depth"]="$DATA_ROOT/egoperceptionmcq"
DATA_ROOTS["adlx_mcq"]="$DATA_ROOT/adlx"
DATA_ROOTS["adlx_descriptions"]="$DATA_ROOT/adlx"

# --- OLLAMA CONFIGURATION (Global) ---
# We set these globally so both the Server AND the Python Client see them
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT:-15000}"
export OLLAMA_MODELS="/path/to/ollama_models"
PROJECT_DIR="/path/to/DnA/viscop"
# Critical: Add lib path for GPU support
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PROJECT_DIR/lib"
SERVER_PID=""

# Check if we need Ollama for ANY of the requested benchmarks
if [[ "$BENCHMARKS" =~ (egoperceptionmcq|egoperceptionmcq_depth|adlx_mcq|adlx_descriptions) ]]; then
    NEEDS_OLLAMA=true
else
    NEEDS_OLLAMA=false
fi

# --- START OLLAMA ONCE (Robustly) ---
if [ "$NEEDS_OLLAMA" = true ]; then
    # Check if already running on this port (e.g. from a previous run)
    if ! curl -s http://127.0.0.1:15000 > /dev/null; then
        echo "Starting Ollama Server..."
        mkdir -p logs
        $PROJECT_DIR/bin/ollama serve > logs/server_${SLURM_JOB_ID:-local}.log 2>&1 &
        SERVER_PID=$!
        
        # Health check loop
        echo "Waiting for Ollama to initialize..."
        for i in {1..20}; do
            if curl -s http://127.0.0.1:15000 > /dev/null; then
                echo "Ollama is UP!"
                break
            fi
            sleep 2
        done
        
        # Verify it didn't crash
        if ! kill -0 $SERVER_PID > /dev/null 2>&1; then
             echo "CRITICAL ERROR: Ollama failed to start. Check logs/server_*.log"
             exit 1
        fi
    else
        echo "Ollama is already running on port 15000."
    fi
fi

# --- BENCHMARK LOOP ---
IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$BENCHMARK]}
    if [ -z "$DATA_ROOT" ]; then
        echo "Error: Data root for benchmark '$BENCHMARK' not defined."
        continue
    fi
    
    echo ">>> Running $BENCHMARK"

    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        evaluation/evaluate_branch_metrics.py \
        --model_path ${MODEL_PATH} \
        --benchmark ${BENCHMARK} \
        --data_root ${DATA_ROOT} \
        --save_path "${SAVE_DIR}/${BENCHMARK}.json" \
        --fps 1 \
        --max_frames 180 \
        --max_visual_tokens 16384 \
        --num_workers 4 \
        --debug ${DEBUG_MODE}
done

# --- CLEANUP ---
if [ ! -z "$SERVER_PID" ]; then
    echo "Stopping Ollama Server..."
    kill $SERVER_PID
fi
