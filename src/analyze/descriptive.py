"""Descriptive statistics for the CoT uncertainty study.

Computes uncertainty rates, transmission rates, suppression rates,
and position distributions by model and task bucket.

Usage:
    python -m src.analyze.descriptive --input data/processed/all_responses.parquet \
        --output data/processed/descriptive_stats.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute all descriptive statistics from the annotated dataset.

    Args:
        df: Annotated DataFrame (from run_annotation.py).

    Returns:
        Nested dict of statistics.
    """
    # Filter to primary samples only
    primary = df[df["sample_index"] == 0].copy()

    stats = {
        "overall": _overall_stats(primary),
        "by_model": _grouped_stats(primary, "model"),
        "by_task_bucket": _grouped_stats(primary, "task_bucket"),
        "by_regime": _grouped_stats(primary, "transparency_regime"),
        "by_model_and_bucket": _cross_stats(primary, "model", "task_bucket"),
        "position_distributions": _position_stats(primary),
    }

    return stats


def _overall_stats(df: pd.DataFrame) -> dict:
    """Compute overall summary statistics."""
    n = len(df)
    if n == 0:
        return {}

    return {
        "n": n,
        "uncertainty_in_artifact_rate": float(df["uncertainty_in_artifact"].mean()),
        "uncertainty_in_answer_rate": float(df["uncertainty_in_answer"].mean()),
        "transmission_rate": float(df["transmission"].mean()),
        "suppression_rate": float(df["suppression"].mean()),
        "mean_artifact_uncertainty_rate": float(df["artifact_uncertainty_rate"].mean()),
        "mean_answer_uncertainty_rate": float(df["answer_uncertainty_rate"].mean()),
        "mean_confidence": float(df["parsed_confidence"].mean()) if "parsed_confidence" in df else None,
        "mean_artifact_token_count": float(df["artifact_token_count"].mean()),
        "revision_event_rate": float(df["artifact_has_revision_event"].mean()),
        "parse_failure_rate": float(df["parse_failure"].mean()),
        "truncation_rate": float(df["truncated"].mean()),
    }


def _grouped_stats(df: pd.DataFrame, group_col: str) -> dict:
    """Compute statistics grouped by a single column."""
    results = {}
    for group_val, group_df in df.groupby(group_col):
        results[str(group_val)] = _overall_stats(group_df)
    return results


def _cross_stats(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Compute statistics grouped by two columns."""
    results = {}
    for (v1, v2), group_df in df.groupby([col1, col2]):
        key = f"{v1}|{v2}"
        results[key] = _overall_stats(group_df)
    return results


def _position_stats(df: pd.DataFrame) -> dict:
    """Compute statistics about uncertainty position distributions."""
    # Filter to responses with uncertainty in artifact
    has_uncertainty = df[df["uncertainty_in_artifact"]].copy()
    if len(has_uncertainty) == 0:
        return {}

    positions = has_uncertainty["artifact_first_uncertainty_position"]
    valid_positions = positions[positions >= 0]

    if len(valid_positions) == 0:
        return {}

    quartile_counts = pd.cut(
        valid_positions, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["Q1", "Q2", "Q3", "Q4"]
    ).value_counts()

    return {
        "n_with_uncertainty": int(len(has_uncertainty)),
        "mean_first_position": float(valid_positions.mean()),
        "median_first_position": float(valid_positions.median()),
        "std_first_position": float(valid_positions.std()),
        "quartile_distribution": {k: int(v) for k, v in quartile_counts.items()},
        "by_model": {
            str(model): {
                "mean_first_position": float(group["artifact_first_uncertainty_position"].mean()),
                "n": int(len(group)),
            }
            for model, group in has_uncertainty.groupby("model")
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute descriptive statistics")
    parser.add_argument("--input", type=Path, required=True, help="Path to annotated parquet file")
    parser.add_argument("--output", type=Path, default=Path("data/processed/descriptive_stats.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    stats = compute_descriptive_stats(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved descriptive stats to %s", args.output)

    # Print highlights
    overall = stats["overall"]
    print("\n=== Descriptive Statistics (Primary Samples) ===")
    print(f"  N: {overall['n']}")
    print(f"  Uncertainty in artifact: {overall['uncertainty_in_artifact_rate']:.1%}")
    print(f"  Uncertainty in answer: {overall['uncertainty_in_answer_rate']:.1%}")
    print(f"  Transmission rate: {overall['transmission_rate']:.1%}")
    print(f"  Suppression rate: {overall['suppression_rate']:.1%}")
    print(f"  Mean artifact uncertainty rate: {overall['mean_artifact_uncertainty_rate']:.2f} per 100 tokens")
    print(f"  Revision event rate: {overall['revision_event_rate']:.1%}")


if __name__ == "__main__":
    main()
