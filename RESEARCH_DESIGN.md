# Research Design: From Thought to Answer
## Measuring Uncertainty Transmission and Suppression Across LLM Reasoning Artifacts

**Target Venue:** NeurIPS 2026 — Evaluations & Datasets Track  
**Abstract Deadline:** May 4, 2026 (AOE)  
**Full Paper Deadline:** May 6, 2026 (AOE)  
**Today:** April 1, 2026 (~5 weeks)

---

## 1. Research Questions

**RQ1:** When models generate reasoning artifacts, how often and where do uncertainty markers appear?

**RQ2:** Does uncertainty in the reasoning artifact predict error, abstention, correction, or hedging in the final answer?

**RQ3:** How much uncertainty survives transmission from reasoning artifact to final answer — and how much is suppressed?

**RQ4:** Do these patterns differ across transparency regimes (raw traces vs. summarized artifacts), models, and task types?

---

## 2. Hypotheses

- **H1:** Uncertainty markers in exposed reasoning artifacts increase on difficult and unanswerable items.
- **H2:** Visible uncertainty in reasoning artifacts predicts lower raw accuracy but better-calibrated abstention.
- **H3:** Suppression rates (uncertainty present in artifact but absent in answer) differ across transparency regimes and model families. *(Descriptive comparison — possible explanations such as RLHF intensity discussed qualitatively, not tested as a direct correlational claim.)*
- **H4:** Earlier uncertainty onset in reasoning artifacts, combined with self-correction cues, predicts better final calibration than late or absent uncertainty.
- **H5:** Behavioral artifact features (hedging, revision events) outperform self-reported numeric confidence for predicting correctness.

---

## 3. Transparency Regimes and Model Selection (5 models)

The study compares uncertainty behavior across two transparency regimes based on what each provider actually exposes through its API.

### Regime 1: Raw Visible Traces (Ollama)

Ollama returns the full reasoning trace via `message.thinking`, separated from `message.content`. This is the only regime where we observe the complete, uncompressed reasoning process.

| Model | Ollama Tag | Size | Partition | Notes |
|-------|-----------|------|-----------|-------|
| DeepSeek-R1-Distill-Llama | `deepseek-r1:70b` | 70B | h200x4 | Distilled R1; full `<think>` block |
| Qwen3 | `qwen3:32b` | 32B | h200x4 | Native thinking mode |

### Regime 2: Summarized Reasoning Artifacts (Production APIs)

All three production providers expose reasoning summaries, not raw internal reasoning tokens. The artifacts vary in format and granularity but are all lossy compressions of the model's internal reasoning process.

| Model | Artifact Format | Notes |
|-------|----------------|-------|
| OpenAI o4-mini | Reasoning summary via `summary` array in reasoning output item (opt-in) | Raw reasoning tokens billed but invisible; summaries are model-generated condensations |
| Gemini 2.5 Flash | Thought summaries via `includeThoughts` / `thinkingConfig` | Google docs explicit: these are summarized thoughts, not raw |
| Claude Sonnet 4.6 | Summarized thinking blocks via adaptive extended thinking | Default API behavior is summarized thinking; full thinking requires special access |

> **Critical design note:** This asymmetry is not a limitation — it is the finding.
> The paper studies uncertainty transmission under the transparency conditions
> that real users and downstream systems actually encounter. The comparison between
> raw traces (Ollama) and summarized artifacts (APIs) is itself a key contribution.

### Scope Reduction Note

Reduced from 6 to 5 models (dropped OpenAI o3) to keep timeline feasible.
Five models provide sufficient coverage: 2 raw-trace, 3 summarized-artifact.

### LLM Judge (Not a study model)

| Model | Purpose |
|-------|---------|
| Llama 4 (70B+ variant, via Ollama on NVWulf) | Uncertainty annotation judge — secondary analysis layer |

---

## 4. Task Design (~500 prompts)

| Bucket | N | Source | Purpose |
|--------|---|--------|---------|
| Answerable reasoning (easy) | 100 | GSM8K easy subset | Low-uncertainty baseline |
| Answerable reasoning (hard) | 100 | MATH Level 4–5 | Difficulty-driven uncertainty |
| Unanswerable / false premise | 100 | SelfAware + AbstentionBench | Epistemic uncertainty, abstention |
| Underspecified | 100 | AbstentionBench subsets | Should trigger hedging |
| Factual QA (verifiable) | 100 | TriviaQA or Natural Questions | Semantic uncertainty estimation |

### Scope Note: Ambiguous Bucket Removed

The original "ambiguous / contested" bucket was dropped. It carried the highest
annotation risk (disputed ground truth, label validity complaints from reviewers)
and the least methodological clarity. The five remaining buckets provide clear
answerable-vs-unanswerable contrasts with established ground truth from existing
benchmarks. If results warrant it, a targeted ambiguity analysis can be added in
a camera-ready revision.

### Selection Criteria
- Balance across domains (math, science, common sense, factual)
- Include ground-truth labels for correctness and answerability
- Avoid tasks where all models trivially succeed or fail
- Pilot 50 prompts first to calibrate difficulty

---

## 5. Prompt Design

### Answer Prompt (provider-agnostic)

The prompt deliberately avoids "think step by step" or similar CoT-inducing
instructions. Reasoning behavior is controlled entirely through provider-native
thinking settings. This prevents the prompt from confounding the study by inducing
extra verbal hedging in the final answer unrelated to the provider's reasoning artifact.

```
Reason carefully about the following question, then answer.

Question: {question}

Respond in the following JSON format:
{
  "final_answer": "<your answer>",
  "abstain": <true or false>,
  "confidence": <0-100>,
  "justification": "<one sentence explaining your confidence>"
}
```

### Provider-Specific Reasoning Configuration

- **Ollama:** `"think": true` in request options — captures `<think>` block in `message.thinking`, final answer in `message.content`
- **Claude Sonnet 4.6:** Adaptive extended thinking with `thinking: {type: "enabled", budget_tokens: 10000}` — returns summarized thinking blocks + text blocks. *(Note: the `type: "enabled"` + `budget_tokens` interface is documented as the current path for Sonnet 4.6. Monitor for deprecation; if Anthropic shifts to effort-only before collection, switch to `thinking: {type: "adaptive", effort: "high"}`. Verify in pilot.)*
- **Gemini 2.5 Flash:** `thinkingConfig: {thinkingBudget: 10000}` — returns thought-summary parts (flagged `thought: true`) alongside answer parts
- **OpenAI o4-mini:** `reasoning: {summary: "detailed"}` in request — returns reasoning summary in the reasoning output item's `summary` array; raw reasoning tokens invisible but billed

### Structured Output

Where providers support it, request structured JSON output to reduce format drift:
- **OpenAI:** Use structured outputs / response format
- **Gemini:** Use response schema constraints
- **Claude:** Use tool_use or structured response prompting
- **Ollama:** Rely on prompt-level JSON instruction with text fallback

All collectors save both the normalized parsed record AND the raw provider response
(see Section 6). If structured parsing fails, the raw payload is preserved for
manual recovery.

---

## 6. Data Collection Schema

Each response stored as a JSON record:

```json
{
  "prompt_id": "str",
  "model": "str",
  "provider": "str (ollama|anthropic|google|openai)",
  "model_version": "str",
  "transparency_regime": "str (raw_trace|summarized_artifact)",
  "task_bucket": "str (easy_reasoning|hard_reasoning|unanswerable|underspecified|factual_qa)",
  "interpolated_prompt": "str — the exact prompt sent after template interpolation",
  "question_text": "str",
  "question_metadata": {
    "source_dataset": "str",
    "ground_truth": "str|null",
    "is_answerable": "bool",
    "difficulty": "str|null"
  },

  "reasoning_artifact": "str — flattened text of reasoning trace or summary",
  "artifact_type": "str (raw_trace|thought_summary|reasoning_summary)",
  "content_blocks": "list — full provider content-block array preserved verbatim",
  "final_answer_raw": "str — the raw final-answer text",

  "parsed_answer": {
    "final_answer": "str",
    "abstain": "bool",
    "confidence_0_100": "int",
    "justification": "str"
  },

  "generation_metadata": {
    "temperature": "float",
    "max_tokens": "int",
    "thinking_budget": "int|null",
    "latency_ms": "int",
    "input_tokens": "int",
    "output_tokens": "int",
    "reasoning_tokens": "int|null — includes invisible reasoning tokens where reported",
    "thinking_tokens": "int|null",
    "timestamp": "str (ISO 8601)",
    "sample_index": "int (0 for primary, 1-4 for repeated sampling)"
  },

  "failure_metadata": {
    "truncated": "bool — response hit max_tokens during reasoning or answer",
    "refusal": "bool — model refused to answer",
    "parse_failure": "bool — structured output parsing failed",
    "incomplete": "bool — response ended without completing the schema",
    "error_message": "str|null"
  },

  "raw_provider_payload": "dict — verbatim API response object"
}
```

### Storage
- One JSONL file per model: `data/raw/{model_name}.jsonl`
- Repeated sampling subset stored alongside primary with `sample_index > 0`
- Raw provider payloads can be large; store separately if needed: `data/raw_payloads/{model_name}/`

### Token Budget Note

OpenAI bills reasoning tokens as output tokens even though they are invisible.
Anthropic charges for full thinking tokens even when only summarized thinking is
returned. Gemini includes thinking tokens in output pricing. Set generous
`max_tokens` / `max_output_tokens` to avoid truncation during reasoning — OpenAI
docs explicitly warn that reasoning models can exhaust the output budget before
producing a visible answer. Pilot must measure actual average token usage per
provider to validate cost estimates.

---

## 7. Repeated Sampling (Semantic Uncertainty)

**Subset:** 100 prompts from the factual QA bucket  
**Samples:** 5 per prompt  
**Models:** Gemini 2.5 Flash, o4-mini, DeepSeek-R1, Qwen3 (cheapest/free options)  
**Temperature:** 0.7 (fixed across models)

This yields 2,000 additional calls (100 x 5 x 4 models). Used to compute:
- Answer dispersion (unique answer clusters)
- Semantic entropy (cluster answers by meaning, compute entropy)
- Compare behavioral uncertainty (sampling) vs. expressed uncertainty (artifact markers)

---

## 8. Uncertainty Annotation

### Layer 1: Lexical Feature Extraction (automated, primary)

Per reasoning artifact and per final answer, extract:

| Feature | Description |
|---------|-------------|
| `hedge_count` | Hedging words: "likely," "probably," "might," "perhaps," "possibly" |
| `uncertainty_statements` | Explicit: "I'm not sure," "I don't know," "unclear" |
| `self_corrections` | Revision cues: "wait," "actually," "let me reconsider," "on second thought" |
| `alternative_hypotheses` | "alternatively," "another possibility," "or it could be" |
| `defeaters` | "unless," "however," "but if" |
| `missing_info_requests` | "I would need," "not enough information," "it depends on" |
| `abstention_markers` | "I cannot answer," "unanswerable," explicit refusal |
| `uncertainty_rate` | Total markers per 100 tokens |
| `first_uncertainty_position` | Normalized position (0.0–1.0) of first marker in artifact |
| `has_revision_event` | Boolean: artifact contains a self-correction sequence |

### Layer 2: Human Validation (required)

Hand-label 150 samples (30 per model) to:
- Validate lexical feature accuracy
- Establish human inter-annotator agreement
- Identify edge cases the automated pipeline misses
- Provide calibration data for the LLM judge

### Layer 3: LLM Judge (secondary analysis, Llama 4 on NVWulf)

For each response, the judge classifies:

1. **Artifact uncertainty level:** none / low / moderate / high
2. **Final answer stance:** assertive / hedged / abstaining
3. **Uncertainty transmission:** transmitted / partially suppressed / fully suppressed
4. **Reasoning quality flags:** contains backtracking, contains error-then-correction, contains contradiction

Judge prompt designed with few-shot examples from the hand-labeled validation set.
Report Cohen's kappa between judge and human labels. If lexical + human layers
already answer the paper's questions, the LLM judge layer is supplementary and
can be deprioritized under time pressure.

---

## 9. Derived Variables (Per Response)

### Binary Metrics

| Variable | Type | Definition |
|----------|------|------------|
| `uncertainty_in_artifact` | bool | Any uncertainty marker in reasoning artifact |
| `uncertainty_in_answer` | bool | Any uncertainty marker in final answer |
| `transmission` | bool | Uncertainty in artifact AND in answer |
| `suppression` | bool | Uncertainty in artifact BUT NOT in answer |

### Continuous Metrics

| Variable | Type | Definition |
|----------|------|------------|
| `uncertainty_rate_artifact` | float | Markers per 100 tokens in artifact |
| `uncertainty_rate_answer` | float | Markers per 100 tokens in answer |
| `attenuation` | float | Standardized difference: `(rate_artifact - rate_answer) / rate_artifact`, computed only when artifact length > 0 AND at least one uncertainty event is present in the artifact. Undefined otherwise. |
| `first_uncertainty_pos` | float | Normalized position of first marker (0.0–1.0) |

### Other Variables

| Variable | Type | Definition |
|----------|------|------------|
| `has_revision` | bool | Contains self-correction event |
| `is_correct` | bool | Answer matches ground truth |
| `is_answerable` | bool | Question has a definite answer |
| `appropriate_behavior` | bool | Correct on answerable OR abstains on unanswerable |
| `self_confidence` | int | Model's self-reported 0–100 confidence |
| `semantic_entropy` | float | From repeated sampling (subset only) |

### Suppression Metric Design Note

The previous `suppression_ratio` (defined as `1 - (answer_rate / artifact_rate)`)
was problematic: undefined when artifact rate is zero, unstable for short traces,
and ill-behaved when answer has more markers than artifact. Replaced with:

1. **Binary suppression** — the primary metric. Simple, interpretable, robust.
2. **Attenuation** — continuous, but only computed on the subset where it is
   well-defined (nonzero artifact length, at least one uncertainty event).

---

## 10. Statistical Analysis Plan

### Layer 1: Descriptive
- Uncertainty rate in artifacts vs. final answers (by model x task bucket)
- Transmission rate and suppression rate (by model x task bucket)
- Suppression rate by transparency regime (raw trace vs. summarized artifact)
- Visualization: heatmaps, uncertainty position distributions, suppression flow diagrams

### Layer 2: Predictive (Mixed-Effects Logistic Regression)

With only 5 models, model is a fixed effect (not random). Task bucket and
transparency regime are also fixed effects. Random intercepts for prompt_id
account for question-level variance.

**Model A — Correctness:**
```
correct ~ uncertainty_in_artifact + uncertainty_in_answer + self_confidence +
          task_bucket + model + (1|prompt_id)
```

**Model B — Appropriate Behavior (correct OR appropriate abstention):**
```
appropriate ~ uncertainty_in_artifact + has_revision + task_bucket +
             transparency_regime + (1|prompt_id)
```

**Model C — Suppression:**
```
suppression ~ model + task_bucket + uncertainty_rate_artifact +
             transparency_regime + (1|prompt_id)
```

### Layer 3: Temporal
```
calibration_error ~ first_uncertainty_pos * task_bucket + has_revision +
                   model + (1|prompt_id)
```

### Layer 4: Semantic Uncertainty Comparison (subset)
- Correlation between semantic entropy and artifact-based uncertainty features
- Does expressed uncertainty in artifacts track behavioral uncertainty from sampling?

### Multiple Comparisons
- FDR correction (Benjamini-Hochberg) across hypothesis tests
- Report effect sizes (odds ratios) with 95% CIs throughout

---

## 11. Evaluation Contribution (Datasets & Benchmarks Track Framing)

The primary contribution is an **evaluation methodology**, not just empirical results:

1. **Uncertainty Suppression Metric:** A formal, reproducible binary metric (with continuous attenuation variant) quantifying how much uncertainty is lost in the artifact-to-answer transition, with clear conditions for well-definedness.

2. **Uncertainty Transmission Taxonomy:** A coding scheme (lexical features + human validation + optional LLM judge) for classifying uncertainty expression in reasoning artifacts, validated against human annotations with reported inter-annotator agreement.

3. **Cross-Provider Evaluation Protocol:** A standardized methodology for comparing reasoning transparency across providers with explicitly different artifact formats, framed around transparency regimes rather than assuming symmetric access.

4. **Curated Task Suite:** 500 prompts with ground-truth labels specifically designed to elicit varying uncertainty types, drawn from existing benchmarks and openly released.

5. **Open Dataset:** All collected artifacts, annotations, derived features, and raw provider payloads released publicly for reproducibility.

---

## 12. Cost Estimate (Provisional — Re-estimate After Pilot)

These estimates assume modest reasoning token usage. Because OpenAI and Anthropic
bill invisible reasoning/thinking tokens at output rates, actual costs depend on
how much internal reasoning each model performs — which is not visible until you
run the pilot. **The pilot (Week 2) must log actual token usage per provider and
produce a revised cost estimate before full collection begins.**

| Component | Calls | Est. Cost (provisional) |
|-----------|-------|------------------------|
| Claude Sonnet 4.6 | 500 | $40–100 (thinking tokens dominate) |
| Gemini 2.5 Flash | 500 + 500 repeated | $15–40 (thinking included in output pricing) |
| OpenAI o4-mini | 500 + 500 repeated | $25–60 (reasoning tokens billed as output) |
| Ollama (NVWulf) | 500 + 1000 repeated | $0 (compute only) |
| Llama 4 judge (NVWulf) | ~2,500 | $0 (compute only) |
| **Total API cost** | | **~$80–200 (provisional)** |

> Final budget confirmed after pilot. If pilot reveals costs exceeding $300,
> reduce repeated sampling subset or drop one API model.

---

## 13. Timeline

| Dates | Week | Milestone | Deliverables |
|-------|------|-----------|--------------|
| Apr 1–7 | 1 | Infrastructure | Collection pipeline (4 provider wrappers), prompt template, NVWulf SLURM scripts, schema validation |
| Apr 8–14 | 2 | Pilot + Task Curation | Pilot 50 prompts across all 5 models; validate parsing, measure actual token usage, re-estimate costs; finalize 500-prompt suite |
| Apr 15–21 | 3 | Full Data Collection | Run all models, collect all artifacts, store raw data; begin lexical annotation |
| Apr 22–28 | 4 | Annotation + Analysis | Complete lexical extraction, hand-label 150 samples, LLM judge (if time permits), fit regressions, error analysis |
| Apr 29–May 4 | 5 | Writing | Draft paper, generate figures, submit abstract May 4 |
| May 4–6 | — | Final push | Polish, supplementary materials, submit full paper May 6 |

### Timeline Risk Mitigation

If behind schedule at Week 3:
- LLM judge becomes appendix-only (lexical + human layers are sufficient)
- Repeated sampling reduced from 100 to 50 prompts
- Paper focuses on core descriptive + Model A/C regressions; temporal analysis deferred

---

## 14. Project Structure

```
cot-analysis/
├── README.md
├── RESEARCH_DESIGN.md          # This file
├── STATUS.md                   # Session continuity
├── requirements.txt
├── Makefile                    # Pipeline orchestration
│
├── data/
│   ├── tasks/                  # Curated prompt suite
│   │   ├── easy_reasoning.jsonl
│   │   ├── hard_reasoning.jsonl
│   │   ├── unanswerable.jsonl
│   │   ├── underspecified.jsonl
│   │   └── factual_qa.jsonl
│   ├── raw/                    # Normalized response records (one JSONL per model)
│   │   ├── deepseek_r1_70b.jsonl
│   │   ├── qwen3_32b.jsonl
│   │   ├── claude_sonnet_4_6.jsonl
│   │   ├── gemini_flash_2_5.jsonl
│   │   └── o4_mini.jsonl
│   ├── raw_payloads/           # Verbatim API responses (one dir per model)
│   │   ├── deepseek_r1_70b/
│   │   ├── qwen3_32b/
│   │   ├── claude_sonnet_4_6/
│   │   ├── gemini_flash_2_5/
│   │   └── o4_mini/
│   └── processed/              # Annotated + derived features
│       └── all_responses.parquet
│
├── src/
│   ├── collect/                # Data collection (separate from analysis)
│   │   ├── __init__.py
│   │   ├── base_collector.py   # Abstract collector interface + schema
│   │   ├── ollama_collector.py
│   │   ├── anthropic_collector.py
│   │   ├── google_collector.py
│   │   ├── openai_collector.py
│   │   └── run_collection.py   # Main collection orchestrator
│   │
│   ├── annotate/               # Uncertainty annotation
│   │   ├── __init__.py
│   │   ├── lexical_features.py # Regex/keyword extraction
│   │   ├── llm_judge.py        # Llama 4 judge pipeline (secondary)
│   │   └── run_annotation.py
│   │
│   ├── analyze/                # Statistical analysis
│   │   ├── __init__.py
│   │   ├── descriptive.py      # Rates, distributions
│   │   ├── regression.py       # Mixed-effects models
│   │   ├── semantic_entropy.py # Repeated sampling analysis
│   │   └── run_analysis.py
│   │
│   └── visualize/              # Figures (completely separate)
│       ├── __init__.py
│       └── figures.py
│
├── scripts/
│   ├── slurm_ollama_collect.sh # NVWulf SLURM job for Ollama collection
│   ├── slurm_judge.sh          # NVWulf SLURM job for LLM judge
│   └── pilot.py                # Quick pilot run (50 prompts, logs token usage)
│
├── notebooks/
│   └── eda.ipynb               # Initial data exploration only
│
└── paper/
    └── neurips_2026/
        ├── main.tex
        └── figures/
```

---

## 15. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Novelty scooped by concurrent work | Medium | Move fast; emphasize evaluation methodology contribution, not just findings |
| Anthropic deprecates `budget_tokens` before collection | Medium | Pilot validates; fallback to adaptive thinking with effort parameter ready |
| OpenAI reasoning output format underdocumented | Medium | Save raw payloads verbatim; abstract parser; verify in pilot |
| Reasoning exhausts `max_tokens` before visible answer | High for some models | Set generous max_tokens (16K+); monitor truncation in failure_metadata; pilot first |
| Models don't produce enough uncertainty markers | Medium | Task suite includes items specifically designed to elicit uncertainty; pilot first |
| LLM judge disagrees with human labels | Low-Medium | Judge is secondary; lexical + human layers are primary |
| Cost exceeds budget due to invisible reasoning tokens | Medium | Pilot measures actual token usage; re-estimate before full run |
| Timeline too tight for NeurIPS deadline | Medium | Scope cuts defined: drop judge to appendix, reduce repeated sampling, narrow regressions |
| Faithfulness objection from reviewers | High | Frame as behavioral/interface-level study of exposed artifacts under real-world transparency conditions, not claims about internal cognition |
| Summarized artifacts too short for meaningful lexical coding | Medium | Pilot will reveal; if summaries are extremely terse, adapt coding scheme or lean on LLM judge for summary regime |

---

## 16. Key Literature

- UQ Survey (KDD 2025): arXiv 2503.15850
- Verbalized Confidence (ICLR 2024): Xiong et al., arXiv 2306.13063
- Linguistic Calibration (ICML 2024): Band et al., arXiv 2404.00474
- CoT-UQ (ACL Findings 2025): arXiv 2502.17214
- CoT Faithfulness (Anthropic 2023): Lanham et al., arXiv 2307.13702
- FUR (EMNLP 2025): Unlearning reasoning steps
- Unfaithful CoT (NeurIPS 2023): Turpin et al., arXiv 2305.04388
- SelfAware (ACL 2023): Yin et al., arXiv 2305.18153
- AbstentionBench (2025): arXiv 2506.09038
- Semantic Entropy (ICLR 2023): Kuhn et al., arXiv 2302.09664
- Cost of Reasoning (2026): arXiv 2603.16728
- Reasoning Models Express Confidence (2025): Yoon et al., arXiv 2505.14489

---

## Appendix A: Changelog

### v2 (April 1, 2026) — Major revision from expert feedback

**Transparency regime taxonomy corrected:**
- Claude Sonnet 4.6, Gemini 2.5 Flash, and OpenAI o4-mini all return summarized
  artifacts, not full reasoning traces. Only Ollama provides raw visible traces.
- Reframed as two regimes: raw_trace (Ollama) vs. summarized_artifact (APIs).

**Model list corrected:**
- DeepSeek-R1 70B is `deepseek-r1:70b` (DeepSeek-R1-Distill-Llama-70B), not "R1-0528 70B". The 0528 upgrade applies to 8B distilled and 671B full only.
- Qwen3 changed from 72B (does not exist on Ollama) to 32B (`qwen3:32b`).
- Dropped OpenAI o3 to reduce scope from 6 to 5 models.

**API configurations corrected:**
- Anthropic: noted `budget_tokens` deprecation risk; adaptive thinking fallback planned.
- OpenAI: corrected to use reasoning summary in reasoning output item's `summary` array, not a `reasoning_content` field.
- Gemini: corrected to summarized thoughts, not raw; pricing corrected (thinking bundled in output).

**Prompt design corrected:**
- Removed "think step by step" instruction to avoid confounding provider-native reasoning.
- Changed to "Reason carefully" + JSON output schema.
- Added structured output support per provider.

**Data schema expanded:**
- Added `raw_provider_payload`, `interpolated_prompt`, `content_blocks`, `failure_metadata`.
- Added `reasoning_tokens` for invisible token tracking.
- Added `transparency_regime` and `artifact_type` fields.

**Methodology corrected:**
- H3 rewritten as descriptive comparison; RLHF correlation claim removed.
- `suppression_ratio` replaced with binary suppression + continuous attenuation (well-defined only when artifact has uncertainty events).
- Model changed from random to fixed effect in regressions (only 5 levels).
- Ambiguous/contested task bucket removed (annotation risk, timeline pressure).
- LLM judge demoted to secondary analysis layer; lexical + human are primary.

**Cost estimate flagged as provisional:**
- Noted that invisible reasoning tokens inflate costs beyond visible output.
- Pilot required to measure actual token usage before committing to full collection.

**Timeline risk mitigation added:**
- Defined specific scope cuts if behind at Week 3.
