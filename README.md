# From Thought to Answer

**Measuring Uncertainty Transmission and Suppression Across LLM Reasoning Artifacts**

This project studies how large language models express uncertainty during reasoning and whether that uncertainty survives into their final answers. We compare models across two transparency regimes: open-weight models that expose full reasoning traces (via Ollama) and production APIs that return summarized reasoning artifacts (OpenAI, Google Gemini, Anthropic Claude).

**Target venue:** NeurIPS 2026 Evaluations & Datasets Track

## Research Questions

1. When models generate reasoning artifacts, how often and where do uncertainty markers appear?
2. Does uncertainty in the reasoning artifact predict error, abstention, or hedging in the final answer?
3. How much uncertainty survives transmission from reasoning artifact to final answer — and how much is suppressed?
4. Do these patterns differ across transparency regimes, models, and task types?

## Models

| Model | Regime | Provider |
|-------|--------|----------|
| DeepSeek-R1-Distill-Llama 70B | Raw trace | Ollama (NVWulf) |
| Qwen3 32B | Raw trace | Ollama (NVWulf) |
| Claude Sonnet 4.6 | Summarized artifact | Anthropic API |
| Gemini 2.5 Flash | Summarized artifact | Google API |
| o4-mini | Summarized artifact | OpenAI API |

## Task Suite (500 prompts)

| Bucket | N | Source |
|--------|---|--------|
| Easy reasoning | 100 | GSM8K |
| Hard reasoning | 100 | MATH Level 4-5 |
| Unanswerable | 100 | SQuAD v2 + BigBench Known Unknowns |
| Underspecified | 100 | CoCoNot + BBQ |
| Factual QA | 100 | TriviaQA |

## Setup

### Local Development (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### NVWulf (GPU Cluster)

```bash
module load cuda12.8/toolkit/12.8.1
conda activate cot-analysis
```

## Pipeline

The pipeline has four stages, each fully independent. Data collection is separated from analysis.

```
curate → collect → annotate → analyze → visualize
```

### 1. Curate Task Suite

```bash
make curate          # Full 500-prompt suite (seed=42)
make curate-dev      # Small 50-prompt dev set
```

Downloads from HuggingFace: GSM8K, MATH, SQuAD v2, BigBench, CoCoNot, BBQ, TriviaQA.

### 2. Collect Responses

```bash
# All models
make collect

# Single model
make collect-model MODEL=ollama:deepseek-r1:70b

# Pilot (50 prompts, reports token usage + cost estimate)
make pilot
```

On NVWulf, submit the SLURM job for Ollama models:

```bash
sbatch scripts/slurm_ollama_collect.sh
```

### 3. Annotate

```bash
make annotate              # Lexical features only (primary)
make annotate-judge        # Lexical + LLM judge (requires Ollama with Llama 4)
```

### 4. Analyze

```bash
make analyze        # Full analysis (descriptive + regression + semantic entropy)
make descriptive    # Descriptive statistics only
make regression     # Mixed-effects models only
make entropy        # Semantic entropy only
```

### 5. Generate Figures

```bash
make visualize      # All publication figures → paper/neurips_2026/figures/
```

### Quick Dev Cycle

```bash
make dev            # pilot → annotate → analyze → visualize
```

## Project Structure

```
cot-analysis/
├── data/
│   ├── tasks/              # Curated prompt suite (version controlled)
│   ├── raw/                # Raw API responses (gitignored)
│   ├── raw_payloads/       # Verbatim API payloads (gitignored)
│   └── processed/          # Annotated parquet + analysis results (gitignored)
├── src/
│   ├── collect/            # Data collection pipeline
│   ├── annotate/           # Uncertainty annotation (lexical + LLM judge)
│   ├── analyze/            # Statistical analysis (descriptive, regression, entropy)
│   └── visualize/          # Publication figure generation
├── scripts/
│   ├── curate_tasks.py     # Task suite curation from HuggingFace
│   ├── pilot.py            # Pilot run with cost estimation
│   ├── slurm_ollama_collect.sh
│   └── slurm_judge.sh
├── paper/neurips_2026/     # LaTeX source + figures
├── RESEARCH_DESIGN.md      # Full research design document
├── STATUS.md               # Session continuity notes
├── Makefile                # Pipeline orchestration
└── requirements.txt
```

## Reproducibility

- All random sampling uses seed 42
- Task suite is deterministic and version controlled in `data/tasks/`
- Raw provider payloads are stored verbatim alongside normalized records
- Token usage and latency logged per response
- Analysis uses FDR correction (Benjamini-Hochberg) for multiple comparisons
