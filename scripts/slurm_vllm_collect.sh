#!/bin/bash
#SBATCH --job-name=cot-vllm-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/cot-vllm-%j.out
#SBATCH --error=logs/cot-vllm-%j.err

# ============================================================
# SLURM job for vLLM-based data collection on NVWulf
#
# GPU count and partition are set at submit time, not in the
# script, so each model gets only what it needs.
#
# Usage:
#   # DeepSeek-R1 70B (needs 2 GPUs):
#   sbatch --partition=h200x4 --gpus=2 scripts/slurm_vllm_collect.sh --export=MODEL=deepseek-r1-70b
#
#   # Qwen3 32B (needs 1 GPU, 2 for comfort):
#   sbatch --partition=h200x4 --gpus=2 scripts/slurm_vllm_collect.sh --export=MODEL=qwen3-32b
#
#   # Pilot only:
#   sbatch --partition=debug-h200x4 --gpus=2 --time=01:00:00 scripts/slurm_vllm_collect.sh --export=MODEL=deepseek-r1-70b,PILOT=1
# ============================================================

set -euo pipefail

# --- Default model if not set via --export ---
MODEL="${MODEL:-deepseek-r1-70b}"
PILOT="${PILOT:-0}"

# --- Model config ---
declare -A HF_MODELS
HF_MODELS[deepseek-r1-70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
HF_MODELS[qwen3-32b]="Qwen/Qwen3-32B"

declare -A TP_SIZE
TP_SIZE[deepseek-r1-70b]=2
TP_SIZE[qwen3-32b]=1

HF_MODEL="${HF_MODELS[$MODEL]}"
TENSOR_PARALLEL="${TP_SIZE[$MODEL]}"

# --- Environment ---
module load cuda12.8/toolkit/12.8.1

SCRATCH="/lustre/nvwulf/scratch/nijjohnson"
PROJECT_DIR="${SCRATCH}/from-thought-to-answer"
export HF_HOME="${SCRATCH}/hf_cache"

# Load HF token
if [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cot-analysis
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd "${PROJECT_DIR}"

VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}/v1"

# --- Start vLLM server ---
echo "[$(date)] Starting vLLM server for ${HF_MODEL} (TP=${TENSOR_PARALLEL})..."
echo "[$(date)] GPUs allocated: ${SLURM_GPUS_ON_NODE:-unknown}"

vllm serve "${HF_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    &

VLLM_PID=$!

# Wait for server to be ready (model loading can take a few minutes)
echo "[$(date)] Waiting for vLLM server to start..."
for i in $(seq 1 300); do
    if curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server ready after ${i}s."
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server process died."
        exit 1
    fi
    if [ $i -eq 300 ]; then
        echo "[$(date)] ERROR: vLLM server failed to start after 300s."
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# --- Run collection ---
COLLECTOR_SPEC="vllm:${MODEL}"

if [ "${PILOT}" = "1" ]; then
    echo "[$(date)] Running PILOT for ${MODEL}..."
    python scripts/pilot.py \
        --tasks-dir data/tasks \
        --output-dir data/raw \
        --models "${COLLECTOR_SPEC}" \
        --log-level INFO
else
    echo "[$(date)] Running FULL COLLECTION for ${MODEL}..."
    python -m src.collect.run_collection \
        --tasks-dir data/tasks \
        --output-dir data/raw \
        --models "${COLLECTOR_SPEC}" \
        --repeated-samples 5 \
        --repeated-bucket factual_qa \
        --log-level INFO
fi

# --- Cleanup ---
echo "[$(date)] Stopping vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "[$(date)] Collection complete for ${MODEL}."
