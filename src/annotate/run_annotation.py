"""Annotation orchestrator — runs lexical features and optional LLM judge.

Usage:
    # Lexical features only (primary):
    python -m src.annotate.run_annotation --input-dir data/raw --output-dir data/processed

    # With LLM judge (secondary):
    python -m src.annotate.run_annotation --input-dir data/raw --output-dir data/processed \
        --judge-model llama4:70b
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.collect.base_collector import ResponseRecord, load_records
from .lexical_features import extract_features, features_to_dict
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


def annotate_records(
    records: list[ResponseRecord],
    judge: LLMJudge | None = None,
) -> pd.DataFrame:
    """Run annotation pipeline on collected records.

    Args:
        records: List of ResponseRecords from data collection.
        judge: Optional LLM judge for secondary annotation.

    Returns:
        DataFrame with all original fields + annotation columns.
    """
    rows = []

    for record in tqdm(records, desc="Annotating"):
        # Base record fields
        row = {
            "prompt_id": record.prompt_id,
            "model": record.model,
            "provider": record.provider.value,
            "model_version": record.model_version,
            "transparency_regime": record.transparency_regime.value,
            "task_bucket": record.task_bucket.value,
            "question_text": record.question_text,
            "source_dataset": record.question_metadata.source_dataset,
            "ground_truth": record.question_metadata.ground_truth,
            "is_answerable": record.question_metadata.is_answerable,
            "difficulty": record.question_metadata.difficulty,
            "artifact_type": record.artifact_type.value,
            "reasoning_artifact": record.reasoning_artifact,
            "final_answer_raw": record.final_answer_raw,
            "parsed_final_answer": record.parsed_answer.final_answer,
            "parsed_abstain": record.parsed_answer.abstain,
            "parsed_confidence": record.parsed_answer.confidence_0_100,
            "parsed_justification": record.parsed_answer.justification,
            "sample_index": record.generation_metadata.sample_index,
            "latency_ms": record.generation_metadata.latency_ms,
            "input_tokens": record.generation_metadata.input_tokens,
            "output_tokens": record.generation_metadata.output_tokens,
            "reasoning_tokens": record.generation_metadata.reasoning_tokens,
            "thinking_tokens": record.generation_metadata.thinking_tokens,
            "truncated": record.failure_metadata.truncated,
            "refusal": record.failure_metadata.refusal,
            "parse_failure": record.failure_metadata.parse_failure,
            "incomplete": record.failure_metadata.incomplete,
        }

        # --- Layer 1: Lexical features ---
        artifact_features = extract_features(record.reasoning_artifact)
        answer_features = extract_features(record.final_answer_raw)

        # Prefix with artifact_ and answer_
        for key, val in features_to_dict(artifact_features).items():
            row[f"artifact_{key}"] = val
        for key, val in features_to_dict(answer_features).items():
            row[f"answer_{key}"] = val

        # --- Derived variables ---
        row["uncertainty_in_artifact"] = artifact_features.total_markers > 0
        row["uncertainty_in_answer"] = answer_features.total_markers > 0
        row["transmission"] = (
            artifact_features.total_markers > 0 and answer_features.total_markers > 0
        )
        row["suppression"] = (
            artifact_features.total_markers > 0 and answer_features.total_markers == 0
        )

        # Attenuation (continuous, only when well-defined)
        if (
            artifact_features.total_markers > 0
            and artifact_features.token_count > 0
            and artifact_features.uncertainty_rate > 0
        ):
            row["attenuation"] = (
                artifact_features.uncertainty_rate - answer_features.uncertainty_rate
            ) / artifact_features.uncertainty_rate
        else:
            row["attenuation"] = None

        # --- Layer 2: LLM judge (if available) ---
        if judge is not None and record.generation_metadata.sample_index == 0:
            judge_result = judge.classify(
                question=record.question_text,
                reasoning_artifact=record.reasoning_artifact,
                final_answer=record.final_answer_raw,
            )
            if "error" not in judge_result:
                row["judge_artifact_uncertainty"] = judge_result.get("artifact_uncertainty")
                row["judge_answer_stance"] = judge_result.get("answer_stance")
                row["judge_transmission"] = judge_result.get("uncertainty_transmission")
                row["judge_has_backtracking"] = judge_result.get("has_backtracking")
                row["judge_has_error_correction"] = judge_result.get("has_error_correction")
                row["judge_has_contradiction"] = judge_result.get("has_contradiction")
            else:
                row["judge_error"] = str(judge_result.get("error", ""))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run annotation pipeline")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with raw JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--judge-model", type=str, default=None, help="Ollama model for LLM judge (e.g. llama4:70b)")
    parser.add_argument("--judge-host", type=str, default=None, help="Ollama host for judge")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load all records
    all_records = []
    for jsonl_path in sorted(args.input_dir.glob("*.jsonl")):
        records = load_records(jsonl_path)
        logger.info("Loaded %d records from %s", len(records), jsonl_path.name)
        all_records.extend(records)

    if not all_records:
        logger.error("No records found in %s", args.input_dir)
        sys.exit(1)

    logger.info("Total records to annotate: %d", len(all_records))

    # Initialize judge if requested
    judge = None
    if args.judge_model:
        logger.info("Initializing LLM judge: %s", args.judge_model)
        judge = LLMJudge(model_name=args.judge_model, host=args.judge_host)

    # Run annotation
    df = annotate_records(all_records, judge=judge)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "all_responses.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved annotated data to %s (%d rows, %d columns)", output_path, len(df), len(df.columns))

    # Also save CSV for inspection
    csv_path = args.output_dir / "all_responses.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV copy to %s", csv_path)

    # Print summary
    print(f"\n=== Annotation Summary ===")
    print(f"  Total records: {len(df)}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Primary samples: {len(df[df['sample_index'] == 0])}")
    print(f"  Repeated samples: {len(df[df['sample_index'] > 0])}")
    print(f"  Uncertainty in artifact: {df['uncertainty_in_artifact'].sum()} ({df['uncertainty_in_artifact'].mean():.1%})")
    print(f"  Uncertainty in answer: {df['uncertainty_in_answer'].sum()} ({df['uncertainty_in_answer'].mean():.1%})")
    print(f"  Transmission: {df['transmission'].sum()} ({df['transmission'].mean():.1%})")
    print(f"  Suppression: {df['suppression'].sum()} ({df['suppression'].mean():.1%})")


if __name__ == "__main__":
    main()
