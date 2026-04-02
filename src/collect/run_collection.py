"""Collection orchestrator — runs data collection across all models.

Usage:
    python -m src.collect.run_collection --tasks-dir data/tasks --output-dir data/raw \
        --models vllm:deepseek-r1-70b vllm:qwen3-32b anthropic google openai

    python -m src.collect.run_collection --tasks-dir data/tasks --output-dir data/raw \
        --models vllm:deepseek-r1-70b --bucket factual_qa --repeated-samples 5
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .base_collector import (
    BaseCollector,
    TaskBucket,
    TaskItem,
    load_tasks,
    save_record,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_collector(model_spec: str, temperature: float = 0.0) -> BaseCollector:
    """Build a collector from a model spec string.

    Imports are lazy so each provider's SDK only needs to be installed
    when that provider is actually used. This allows running vLLM models
    on the cluster without installing anthropic/google-genai/etc.

    Formats:
        vllm:<model_name>            e.g. vllm:deepseek-r1-70b
        vllm:<model_name>@<url>      e.g. vllm:deepseek-r1-70b@http://host:8000/v1
        anthropic                    uses default Claude Sonnet 4.6
        anthropic:<model_name>       e.g. anthropic:claude-sonnet-4-6-20260401
        google                       uses default Gemini 2.5 Flash
        google:<model_name>          e.g. google:gemini-2.5-flash
        openai                       uses default o4-mini
        openai:<model_name>          e.g. openai:o4-mini
    """
    parts = model_spec.split(":", 1)
    provider = parts[0]

    if provider == "vllm":
        from .vllm_collector import VLLMCollector

        rest = parts[1] if len(parts) > 1 else "deepseek-r1-70b"
        # Support vllm:model@url format
        if "@" in rest:
            model_name, base_url = rest.split("@", 1)
        else:
            model_name = rest
            base_url = "http://localhost:8000/v1"
        return VLLMCollector(
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
        )

    elif provider == "anthropic":
        from .anthropic_collector import AnthropicCollector

        model_name = parts[1] if len(parts) > 1 else "claude-sonnet-4-6-20260401"
        return AnthropicCollector(model_name=model_name, temperature=1.0)

    elif provider == "google":
        from .google_collector import GoogleCollector

        model_name = parts[1] if len(parts) > 1 else "gemini-2.5-flash"
        return GoogleCollector(model_name=model_name, temperature=temperature)

    elif provider == "openai":
        from .openai_collector import OpenAICollector

        model_name = parts[1] if len(parts) > 1 else "o4-mini"
        return OpenAICollector(model_name=model_name, temperature=1.0)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Collection runner
# ---------------------------------------------------------------------------

def run_collection(
    tasks: list[TaskItem],
    collector: BaseCollector,
    output_dir: Path,
    repeated_samples: int = 0,
    repeated_bucket: TaskBucket | None = None,
    repeated_temperature: float = 0.7,
) -> dict[str, int]:
    """Run collection for a single model across all tasks.

    Args:
        tasks: List of task items to collect.
        collector: The provider-specific collector.
        output_dir: Directory to write JSONL output.
        repeated_samples: Number of additional samples for repeated sampling (0 = none).
        repeated_bucket: Only do repeated sampling for this task bucket.
        repeated_temperature: Temperature for repeated sampling.

    Returns:
        Summary stats dict.
    """
    stats = {"total": 0, "success": 0, "parse_fail": 0, "error": 0, "repeated": 0}

    model_label = f"{collector.provider.value}:{collector.model_name}"
    logger.info("Starting collection for %s (%d tasks)", model_label, len(tasks))

    for task in tqdm(tasks, desc=model_label):
        # Primary sample (temperature=0 or provider default)
        record = collector.collect(task, sample_index=0)
        save_record(record, output_dir)
        stats["total"] += 1

        if record.failure_metadata.error_message:
            stats["error"] += 1
        elif record.failure_metadata.parse_failure:
            stats["parse_fail"] += 1
        else:
            stats["success"] += 1

        # Repeated sampling for semantic uncertainty
        if repeated_samples > 0 and (
            repeated_bucket is None or task.task_bucket == repeated_bucket
        ):
            original_temp = collector.temperature
            collector.temperature = repeated_temperature

            for sample_idx in range(1, repeated_samples + 1):
                sample_record = collector.collect(task, sample_index=sample_idx)
                save_record(sample_record, output_dir)
                stats["repeated"] += 1

            collector.temperature = original_temp

    logger.info(
        "Collection complete for %s: %d total, %d success, %d parse_fail, %d error, %d repeated",
        model_label,
        stats["total"],
        stats["success"],
        stats["parse_fail"],
        stats["error"],
        stats["repeated"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run data collection for CoT uncertainty study")
    parser.add_argument("--tasks-dir", type=Path, required=True, help="Directory with task JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--models", nargs="+", required=True, help="Model specs (e.g. ollama:deepseek-r1:70b)")
    parser.add_argument("--bucket", type=str, default=None, help="Only run tasks from this bucket")
    parser.add_argument("--repeated-samples", type=int, default=0, help="Number of repeated samples (for semantic uncertainty)")
    parser.add_argument("--repeated-bucket", type=str, default="factual_qa", help="Only repeat-sample this bucket")
    parser.add_argument("--repeated-temperature", type=float, default=0.7, help="Temperature for repeated sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for primary sampling")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load tasks
    tasks = load_tasks(args.tasks_dir)
    if not tasks:
        logger.error("No tasks found in %s", args.tasks_dir)
        sys.exit(1)
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks_dir)

    # Filter by bucket if specified
    if args.bucket:
        bucket = TaskBucket(args.bucket)
        tasks = [t for t in tasks if t.task_bucket == bucket]
        logger.info("Filtered to %d tasks in bucket '%s'", len(tasks), args.bucket)

    # Parse repeated bucket
    repeated_bucket = TaskBucket(args.repeated_bucket) if args.repeated_bucket else None

    # Run collection for each model
    all_stats = {}
    for model_spec in args.models:
        collector = build_collector(model_spec, temperature=args.temperature)
        stats = run_collection(
            tasks=tasks,
            collector=collector,
            output_dir=args.output_dir,
            repeated_samples=args.repeated_samples,
            repeated_bucket=repeated_bucket,
            repeated_temperature=args.repeated_temperature,
        )
        all_stats[model_spec] = stats

    # Print summary
    print("\n=== Collection Summary ===")
    for model_spec, stats in all_stats.items():
        print(f"  {model_spec}: {stats}")


if __name__ == "__main__":
    main()
