"""Pilot script — run 50 prompts across all models to validate pipeline.

This script:
1. Selects 10 prompts per task bucket (50 total)
2. Runs each through all configured models
3. Reports: parse success rate, avg token usage, avg latency, cost estimate
4. Saves results to data/raw/ for inspection

Usage:
    python scripts/pilot.py --tasks-dir data/tasks --output-dir data/raw

    # Single model pilot:
    python scripts/pilot.py --tasks-dir data/tasks --models ollama:deepseek-r1:70b
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect.base_collector import TaskBucket, load_tasks, save_record
from src.collect.run_collection import build_collector

logger = logging.getLogger(__name__)

# Cost per million tokens (input/output) — provisional
COST_TABLE = {
    "anthropic": {"input": 3.0, "output": 15.0, "thinking": 15.0},
    "google": {"input": 0.15, "output": 3.50, "thinking": 3.50},
    "openai": {"input": 1.10, "output": 4.40, "thinking": 4.40},
    "ollama": {"input": 0.0, "output": 0.0, "thinking": 0.0},
}

DEFAULT_MODELS = [
    "ollama:deepseek-r1:70b",
    "ollama:qwen3:32b",
    "anthropic",
    "google",
    "openai",
]

PROMPTS_PER_BUCKET = 10
SEED = 42


def select_pilot_subset(tasks_dir: Path, per_bucket: int = PROMPTS_PER_BUCKET) -> list:
    """Select a balanced subset of tasks for the pilot."""
    all_tasks = load_tasks(tasks_dir)
    if not all_tasks:
        logger.error("No tasks found in %s", tasks_dir)
        sys.exit(1)

    by_bucket = defaultdict(list)
    for task in all_tasks:
        by_bucket[task.task_bucket].append(task)

    random.seed(SEED)
    selected = []
    for bucket in TaskBucket:
        bucket_tasks = by_bucket.get(bucket, [])
        if not bucket_tasks:
            logger.warning("No tasks found for bucket '%s'", bucket.value)
            continue
        n = min(per_bucket, len(bucket_tasks))
        selected.extend(random.sample(bucket_tasks, n))

    logger.info("Selected %d pilot tasks across %d buckets", len(selected), len(by_bucket))
    return selected


def estimate_full_run_cost(
    model_spec: str,
    provider: str,
    avg_input_tokens: float,
    avg_output_tokens: float,
    avg_reasoning_tokens: float,
    total_prompts: int = 500,
    repeated_samples: int = 500,
) -> float:
    """Estimate full-run cost from pilot averages."""
    costs = COST_TABLE.get(provider, {"input": 0, "output": 0, "thinking": 0})
    total_calls = total_prompts + repeated_samples

    input_cost = (avg_input_tokens * total_calls / 1_000_000) * costs["input"]
    output_cost = (avg_output_tokens * total_calls / 1_000_000) * costs["output"]
    thinking_cost = (avg_reasoning_tokens * total_calls / 1_000_000) * costs["thinking"]

    return input_cost + output_cost + thinking_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot run for CoT uncertainty study")
    parser.add_argument("--tasks-dir", type=Path, default=Path("data/tasks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--per-bucket", type=int, default=PROMPTS_PER_BUCKET)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tasks = select_pilot_subset(args.tasks_dir, per_bucket=args.per_bucket)
    if not tasks:
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"PILOT RUN: {len(tasks)} tasks x {len(args.models)} models")
    print(f"{'='*70}\n")

    for model_spec in args.models:
        print(f"\n--- {model_spec} ---")

        try:
            collector = build_collector(model_spec)
        except Exception as e:
            print(f"  SKIPPED: could not initialize collector: {e}")
            continue

        results = []
        success_count = 0
        parse_fail_count = 0
        error_count = 0

        for task in tasks:
            record = collector.collect(task, sample_index=0)
            save_record(record, args.output_dir)
            results.append(record)

            if record.failure_metadata.error_message:
                error_count += 1
            elif record.failure_metadata.parse_failure:
                parse_fail_count += 1
            else:
                success_count += 1

        # Compute stats
        latencies = [r.generation_metadata.latency_ms for r in results if r.generation_metadata.latency_ms > 0]
        input_tokens = [r.generation_metadata.input_tokens for r in results if r.generation_metadata.input_tokens > 0]
        output_tokens = [r.generation_metadata.output_tokens for r in results if r.generation_metadata.output_tokens > 0]
        reasoning_tokens = [
            r.generation_metadata.reasoning_tokens or r.generation_metadata.thinking_tokens or 0
            for r in results
        ]
        artifact_lengths = [len(r.reasoning_artifact) for r in results if r.reasoning_artifact]
        truncated = sum(1 for r in results if r.failure_metadata.truncated)

        avg_input = sum(input_tokens) / len(input_tokens) if input_tokens else 0
        avg_output = sum(output_tokens) / len(output_tokens) if output_tokens else 0
        avg_reasoning = sum(reasoning_tokens) / len(reasoning_tokens) if reasoning_tokens else 0

        # Cost estimate
        provider = collector.provider.value
        estimated_cost = estimate_full_run_cost(
            model_spec, provider, avg_input, avg_output, avg_reasoning,
        )

        print(f"  Success: {success_count}/{len(results)}  Parse fail: {parse_fail_count}  Error: {error_count}  Truncated: {truncated}")
        print(f"  Avg latency: {sum(latencies)/len(latencies):.0f}ms" if latencies else "  Avg latency: N/A")
        print(f"  Avg input tokens: {avg_input:.0f}")
        print(f"  Avg output tokens: {avg_output:.0f}")
        print(f"  Avg reasoning/thinking tokens: {avg_reasoning:.0f}")
        print(f"  Avg artifact length (chars): {sum(artifact_lengths)/len(artifact_lengths):.0f}" if artifact_lengths else "  Avg artifact length: N/A")
        print(f"  Estimated full-run cost (500 prompts + 500 repeated): ${estimated_cost:.2f}")

    print(f"\n{'='*70}")
    print("Pilot complete. Inspect data/raw/ for output files.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
