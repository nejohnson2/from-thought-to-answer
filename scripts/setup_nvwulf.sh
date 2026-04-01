#!/bin/bash
# ============================================================
# NVWulf setup script
#
# Run this once after cloning the repo on the cluster:
#   bash scripts/setup_nvwulf.sh
# ============================================================

set -euo pipefail

SCRATCH="/lustre/nvwulf/scratch/nijjohnson"
PROJECT_DIR="${SCRATCH}/cot-analysis"
HF_CACHE="${SCRATCH}/hf_cache"

echo "[$(date)] Setting up cot-analysis on NVWulf..."

# --- Verify we're in the right place ---
if [ ! -f "RESEARCH_DESIGN.md" ]; then
    echo "ERROR: Run this script from the project root directory."
    exit 1
fi

# --- Load CUDA ---
module load cuda12.8/toolkit/12.8.0

# --- Create conda environment ---
echo "[$(date)] Creating conda environment 'cot-analysis'..."
if conda info --envs | grep -q "cot-analysis"; then
    echo "  Environment already exists, updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo "[$(date)] Activating environment..."
conda activate cot-analysis

# --- Set up HuggingFace cache ---
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# Load HF token if available
if [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
    echo "[$(date)] HF token loaded."
fi

# --- Create data directories ---
mkdir -p data/raw data/raw_payloads data/processed
mkdir -p "${SCRATCH}/logs"

# --- Download models (optional, can also happen at serve time) ---
echo ""
echo "Setup complete. To pre-download models, run:"
echo ""
echo "  conda activate cot-analysis"
echo "  export HF_HOME=${HF_CACHE}"
echo "  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
echo "  huggingface-cli download Qwen/Qwen3-32B"
echo ""
echo "To start a collection run, submit the SLURM jobs:"
echo ""
echo "  sbatch scripts/slurm_vllm_collect.sh --export=MODEL=deepseek-r1-70b"
echo "  sbatch scripts/slurm_vllm_collect.sh --export=MODEL=qwen3-32b"
echo ""
echo "[$(date)] Done."
