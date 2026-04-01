.PHONY: all curate collect annotate analyze visualize pilot clean dev help

# Directories
TASKS_DIR = data/tasks
RAW_DIR = data/raw
PROCESSED_DIR = data/processed
FIGURES_DIR = paper/neurips_2026/figures

# Default models for collection
MODELS = ollama:deepseek-r1:70b ollama:qwen3:32b anthropic google openai

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
# Development / testing
# ============================================================

dev: pilot annotate analyze visualize

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated data (preserves tasks)
clean:
	rm -rf $(RAW_DIR)/*.jsonl
	rm -rf data/raw_payloads/*/*.json
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png

help:
	@echo "Available targets:"
	@echo "  all            - Full pipeline: collect → annotate → analyze → visualize"
	@echo "  collect        - Run data collection across all models"
	@echo "  collect-model  - Collect for one model (MODEL=ollama:deepseek-r1:70b)"
	@echo "  pilot          - Run 50-prompt pilot across all models"
	@echo "  pilot-model    - Pilot for one model (MODEL=ollama:deepseek-r1:70b)"
	@echo "  annotate       - Run lexical annotation"
	@echo "  annotate-judge - Run lexical + LLM judge annotation"
	@echo "  analyze        - Run full analysis pipeline"
	@echo "  descriptive    - Descriptive statistics only"
	@echo "  regression     - Regression models only"
	@echo "  entropy        - Semantic entropy only"
	@echo "  visualize      - Generate all figures"
	@echo "  dev            - Quick dev cycle: pilot → annotate → analyze → visualize"
	@echo "  install        - Install Python dependencies"
	@echo "  clean          - Remove generated data (preserves tasks)"
