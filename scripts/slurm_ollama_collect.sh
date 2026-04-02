#!/bin/bash
#SBATCH --job-name=cot-ollama-collect
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/cot-collect-%j.out
#SBATCH --error=logs/cot-collect-%j.err

# ============================================================
# SLURM job for Ollama-based data collection on NVWulf
#
# This script:
# 1. Starts an Ollama server on the allocated GPU node
# 2. Pulls the required models
# 3. Runs data collection for both Ollama models
# ============================================================

set -euo pipefail

# --- Environment setup ---
module load cuda12.8/toolkit/12.8.1

SCRATCH="/lustre/nvwulf/scratch/nijjohnson"
PROJECT_DIR="${SCRATCH}/from-thought-to-answer"
OLLAMA_DIR="${SCRATCH}/ollama"
LOG_DIR="${SCRATCH}/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OLLAMA_DIR}"

# Set Ollama environment
export OLLAMA_MODELS="${OLLAMA_DIR}/models"
export OLLAMA_HOST="127.0.0.1:11434"

# Activate conda environment
conda activate cot-analysis

# --- Start Ollama server ---
echo "[$(date)] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 10  # Wait for server to start

# Verify server is running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "[$(date)] ERROR: Ollama server failed to start"
    exit 1
fi
echo "[$(date)] Ollama server running (PID: ${OLLAMA_PID})"

# --- Pull models (if not already present) ---
echo "[$(date)] Pulling models..."
ollama pull deepseek-r1:70b
ollama pull qwen3:32b

# --- Run collection: DeepSeek-R1 ---
echo "[$(date)] Starting collection: deepseek-r1:70b"
python -m src.collect.run_collection \
    --tasks-dir "${PROJECT_DIR}/data/tasks" \
    --output-dir "${PROJECT_DIR}/data/raw" \
    --models "ollama:deepseek-r1:70b" \
    --repeated-samples 5 \
    --repeated-bucket factual_qa \
    --log-level INFO

# --- Run collection: Qwen3 ---
echo "[$(date)] Starting collection: qwen3:32b"
python -m src.collect.run_collection \
    --tasks-dir "${PROJECT_DIR}/data/tasks" \
    --output-dir "${PROJECT_DIR}/data/raw" \
    --models "ollama:qwen3:32b" \
    --repeated-samples 5 \
    --repeated-bucket factual_qa \
    --log-level INFO

# --- Cleanup ---
echo "[$(date)] Stopping Ollama server..."
kill ${OLLAMA_PID} 2>/dev/null || true
wait ${OLLAMA_PID} 2>/dev/null || true

echo "[$(date)] Collection complete."
