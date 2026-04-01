# Status — CoT Uncertainty Analysis

**Date:** 2026-04-01  
**Target:** NeurIPS 2026 Evaluations & Datasets Track (May 4/6 deadline)

## What Was Completed

### Research Design (v2)
- Full research design finalized in `RESEARCH_DESIGN.md`
- Literature review completed, key papers verified
- Feasibility assessed: APIs, models, costs validated
- Expert feedback incorporated (transparency regimes corrected, metrics fixed, scope cuts made)

### Infrastructure (All Modules Built + Smoke Tested)
- **Data collection pipeline** (`src/collect/`):
  - `base_collector.py` — Pydantic schema, abstract collector, prompt template, storage
  - `ollama_collector.py` — Full thinking traces via Ollama
  - `anthropic_collector.py` — Claude extended thinking (summarized)
  - `google_collector.py` — Gemini thought summaries
  - `openai_collector.py` — OpenAI reasoning summaries (Responses API)
  - `run_collection.py` — CLI orchestrator with repeated sampling support
- **Annotation pipeline** (`src/annotate/`):
  - `lexical_features.py` — Regex-based uncertainty marker extraction
  - `llm_judge.py` — Llama 4 judge with few-shot examples
  - `run_annotation.py` — Orchestrator producing annotated parquet
- **Analysis pipeline** (`src/analyze/`):
  - `descriptive.py` — Rates, distributions, cross-tabulations
  - `regression.py` — Mixed-effects models A/B/C + temporal
  - `semantic_entropy.py` — Repeated sampling entropy + correlations
  - `run_analysis.py` — Full analysis orchestrator
- **Visualization** (`src/visualize/figures.py`) — 5 publication-ready figures
- **SLURM scripts** — `slurm_ollama_collect.sh`, `slurm_judge.sh`
- **Pilot script** — `scripts/pilot.py` with cost estimation
- **Makefile** — Full pipeline orchestration

## What's Left

### Week 2 (Apr 8-14): Pilot + Task Curation
- [ ] Curate 500-prompt task suite from GSM8K, MATH, SelfAware, AbstentionBench, TriviaQA
- [ ] Run pilot (50 prompts) across all 5 models
- [ ] Validate parsing, measure actual token usage, re-estimate costs
- [ ] Set up `.env` with API keys

### Week 3 (Apr 15-21): Full Data Collection
- [ ] Run full collection (Ollama models on NVWulf, APIs locally)
- [ ] Begin lexical annotation

### Week 4 (Apr 22-28): Annotation + Analysis
- [ ] Complete annotation, hand-label 150 samples
- [ ] Run statistical analysis, generate figures

### Week 5 (Apr 29-May 6): Writing + Submission
- [ ] Draft and submit paper

## Known Issues
- Anthropic `budget_tokens` may be deprecated; adaptive thinking fallback ready
- OpenAI Responses API field structure needs pilot verification
- Cost estimate is provisional ($80-200); pilot will produce real numbers
- `is_correct` column requires task-specific scoring logic (not yet implemented)
