#!/bin/bash
#SBATCH --job-name=cot-llm-judge
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/cot-judge-%j.out
#SBATCH --error=logs/cot-judge-%j.err

# ============================================================
# SLURM job for LLM judge annotation via vLLM on NVWulf
# Serves Llama 4 70B and runs the annotation pipeline
# ============================================================

set -euo pipefail

module load cuda12.8/toolkit/12.8.1

SCRATCH="/lustre/nvwulf/scratch/nijjohnson"
PROJECT_DIR="${SCRATCH}/cot-analysis"
export HF_HOME="${SCRATCH}/hf_cache"

if [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
fi

conda activate cot-analysis
cd "${PROJECT_DIR}"

JUDGE_MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}/v1"

# Start vLLM server for judge model
echo "[$(date)] Starting vLLM server for judge model..."
vllm serve "${JUDGE_MODEL}" \
    --tensor-parallel-size 4 \
    --port "${VLLM_PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests \
    &

VLLM_PID=$!

# Wait for server
for i in $(seq 1 120); do
    if curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
        echo "[$(date)] vLLM judge server ready after ${i}s."
        break
    fi
    if [ $i -eq 120 ]; then
        echo "[$(date)] ERROR: vLLM server failed to start."
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Run annotation with vLLM-based judge
echo "[$(date)] Starting LLM judge annotation..."
python -m src.annotate.run_annotation \
    --input-dir data/raw \
    --output-dir data/processed \
    --judge-model "${JUDGE_MODEL}" \
    --judge-host "${VLLM_URL}" \
    --log-level INFO

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "[$(date)] Judge annotation complete."
