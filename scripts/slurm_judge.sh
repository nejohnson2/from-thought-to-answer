#!/bin/bash
#SBATCH --job-name=cot-llm-judge
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/cot-judge-%j.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/cot-judge-%j.err

# ============================================================
# SLURM job for LLM judge annotation on NVWulf
# Runs Llama 4 via Ollama to classify uncertainty in responses
# ============================================================

set -euo pipefail

module load cuda12.8/toolkit/12.8.0

SCRATCH="/lustre/nvwulf/scratch/nijjohnson"
PROJECT_DIR="${SCRATCH}/cot-analysis"
OLLAMA_DIR="${SCRATCH}/ollama"

export OLLAMA_MODELS="${OLLAMA_DIR}/models"
export OLLAMA_HOST="127.0.0.1:11434"

conda activate cot-analysis

# Start Ollama server
echo "[$(date)] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 10

if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "[$(date)] ERROR: Ollama server failed to start"
    exit 1
fi

# Pull judge model
echo "[$(date)] Pulling Llama 4 judge model..."
ollama pull llama4:70b

# Run annotation
echo "[$(date)] Starting LLM judge annotation..."
python -m src.annotate.run_annotation \
    --input-dir "${PROJECT_DIR}/data/raw" \
    --output-dir "${PROJECT_DIR}/data/processed" \
    --judge-model "llama4:70b" \
    --log-level INFO

# Cleanup
kill ${OLLAMA_PID} 2>/dev/null || true
wait ${OLLAMA_PID} 2>/dev/null || true

echo "[$(date)] Judge annotation complete."
