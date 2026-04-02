.PHONY: all curate collect annotate analyze visualize pilot clean dev help

# Directories
TASKS_DIR = data/tasks
RAW_DIR = data/raw
PROCESSED_DIR = data/processed
FIGURES_DIR = paper/neurips_2026/figures

# Default models for collection
# vLLM models run on NVWulf, API models run locally
MODELS_VLLM = vllm:deepseek-r1-70b vllm:qwen3-32b
MODELS_API = anthropic google openai
MODELS = $(MODELS_VLLM) $(MODELS_API)

# ============================================================
# Full pipeline
# ============================================================

all: collect annotate analyze visualize

# ============================================================
# Task curation (download datasets + random sample)
# ============================================================

curate:
	python scripts/curate_tasks.py --per-bucket 100 --seed 42

# Small dev set
curate-dev:
	python scripts/curate_tasks.py --per-bucket 10 --seed 42

# ============================================================
# Data collection (run separately from analysis)
# ============================================================

collect:
	python -m src.collect.run_collection \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--models $(MODELS) \
		--repeated-samples 5 \
		--repeated-bucket factual_qa \
		--log-level INFO

# Collect a single model (use: make collect-model MODEL=ollama:deepseek-r1:70b)
collect-model:
	python -m src.collect.run_collection \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--models $(MODEL) \
		--repeated-samples 5 \
		--repeated-bucket factual_qa \
		--log-level INFO

# ============================================================
# Pilot (50 prompts, validates pipeline)
# ============================================================

pilot:
	python scripts/pilot.py \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--log-level INFO

# Single-model pilot
pilot-model:
	python scripts/pilot.py \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--models $(MODEL) \
		--log-level INFO

# ============================================================
# Annotation (lexical features + optional LLM judge)
# ============================================================

annotate:
	python -m src.annotate.run_annotation \
		--input-dir $(RAW_DIR) \
		--output-dir $(PROCESSED_DIR) \
		--log-level INFO

# With LLM judge (requires Ollama with Llama 4)
annotate-judge:
	python -m src.annotate.run_annotation \
		--input-dir $(RAW_DIR) \
		--output-dir $(PROCESSED_DIR) \
		--judge-model llama4:70b \
		--log-level INFO

# ============================================================
# Analysis (reads from processed, writes results)
# ============================================================

analyze:
	python -m src.analyze.run_analysis \
		--input $(PROCESSED_DIR)/all_responses.parquet \
		--output-dir $(PROCESSED_DIR) \
		--log-level INFO

# Individual analysis modules
descriptive:
	python -m src.analyze.descriptive \
		--input $(PROCESSED_DIR)/all_responses.parquet \
		--output $(PROCESSED_DIR)/descriptive_stats.json

regression:
	python -m src.analyze.regression \
		--input $(PROCESSED_DIR)/all_responses.parquet \
		--output $(PROCESSED_DIR)/regression_results.json

entropy:
	python -m src.analyze.semantic_entropy \
		--input $(PROCESSED_DIR)/all_responses.parquet \
		--output $(PROCESSED_DIR)/semantic_entropy.json

# ============================================================
# Visualization (reads saved results, generates figures)
# ============================================================

visualize:
	python -m src.visualize.figures \
		--stats-dir $(PROCESSED_DIR) \
		--output-dir $(FIGURES_DIR) \
		--log-level INFO

# ============================================================
# Split collection: API models locally, vLLM models on cluster
# ============================================================

# Run API models only (locally)
collect-api:
	python -m src.collect.run_collection \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--models $(MODELS_API) \
		--repeated-samples 5 \
		--repeated-bucket factual_qa \
		--log-level INFO

# Submit vLLM collection jobs to NVWulf (run from cluster)
collect-cluster:
	sbatch scripts/slurm_vllm_collect.sh --export=MODEL=deepseek-r1-70b
	sbatch scripts/slurm_vllm_collect.sh --export=MODEL=qwen3-32b

# Pilot API models only (locally)
pilot-api:
	python scripts/pilot.py \
		--tasks-dir $(TASKS_DIR) \
		--output-dir $(RAW_DIR) \
		--models $(MODELS_API) \
		--log-level INFO

# ============================================================
# Data transfer (sync between local and cluster)
# ============================================================

NVWULF_HOST ?= nvwulf
NVWULF_PROJECT = /lustre/nvwulf/scratch/nijjohnson/from-thought-to-answer

# Push task suite + code to cluster
push-cluster:
	rsync -avz --exclude='.venv' --exclude='data/raw' --exclude='data/raw_payloads' \
		--exclude='data/processed' --exclude='__pycache__' --exclude='.env' \
		./ $(NVWULF_HOST):$(NVWULF_PROJECT)/

# Pull collected data from cluster
pull-cluster:
	rsync -avz $(NVWULF_HOST):$(NVWULF_PROJECT)/data/raw/ data/raw/
	rsync -avz $(NVWULF_HOST):$(NVWULF_PROJECT)/data/raw_payloads/ data/raw_payloads/

# ============================================================
# Development / testing
# ============================================================

dev: pilot annotate analyze visualize

# Install dependencies (local)
install:
	pip install -r requirements.txt

# Setup cluster environment
install-cluster:
	bash scripts/setup_nvwulf.sh

# Clean generated data (preserves tasks)
clean:
	rm -rf $(RAW_DIR)/*.jsonl
	rm -rf data/raw_payloads/*/*.json
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Task curation:"
	@echo "    curate         - Full 500-prompt suite"
	@echo "    curate-dev     - Small 50-prompt dev set"
	@echo ""
	@echo "  Collection:"
	@echo "    collect        - All models (requires vLLM + API keys)"
	@echo "    collect-api    - API models only (run locally)"
	@echo "    collect-model  - Single model (MODEL=vllm:deepseek-r1-70b)"
	@echo "    collect-cluster - Submit vLLM SLURM jobs (run from cluster)"
	@echo "    pilot          - 50-prompt pilot, all models"
	@echo "    pilot-api      - 50-prompt pilot, API models only"
	@echo "    pilot-model    - 50-prompt pilot, single model"
	@echo ""
	@echo "  Cluster sync:"
	@echo "    push-cluster   - Push code + tasks to NVWulf"
	@echo "    pull-cluster   - Pull collected data from NVWulf"
	@echo ""
	@echo "  Annotation:"
	@echo "    annotate       - Lexical features (primary)"
	@echo "    annotate-judge - Lexical + LLM judge"
	@echo ""
	@echo "  Analysis:"
	@echo "    analyze        - Full analysis pipeline"
	@echo "    descriptive    - Descriptive stats only"
	@echo "    regression     - Regression models only"
	@echo "    entropy        - Semantic entropy only"
	@echo ""
	@echo "  Visualization:"
	@echo "    visualize      - Generate all figures"
	@echo ""
	@echo "  Setup:"
	@echo "    install        - Install local dependencies"
	@echo "    install-cluster - Setup NVWulf conda env"
	@echo "    dev            - Quick dev cycle"
	@echo "    clean          - Remove generated data"
