"""Analysis orchestrator — runs all analysis modules in sequence.

Usage:
    python -m src.analyze.run_analysis --input data/processed/all_responses.parquet \
        --output-dir data/processed
"""

import argparse
import logging
import sys
from pathlib import Path

from .descriptive import compute_descriptive_stats
from .regression import prepare_data, fit_model_a, fit_model_b, fit_model_c, fit_temporal_model, apply_fdr_correction
from .semantic_entropy import compute_semantic_entropy, analyze_entropy_correlations

import json
import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Descriptive statistics
    print("\n=== Phase 1: Descriptive Statistics ===")
    desc_stats = compute_descriptive_stats(df)
    desc_path = args.output_dir / "descriptive_stats.json"
    with open(desc_path, "w") as f:
        json.dump(desc_stats, f, indent=2, default=str)
    logger.info("Saved descriptive stats to %s", desc_path)

    # 2. Regression models
    print("\n=== Phase 2: Regression Models ===")
    prepared = prepare_data(df)
    regression_results = {
        "model_a_correctness": fit_model_a(prepared),
        "model_b_appropriate": fit_model_b(prepared),
        "model_c_suppression": fit_model_c(prepared),
        "temporal_calibration": fit_temporal_model(prepared),
    }
    regression_results = apply_fdr_correction(regression_results)

    reg_path = args.output_dir / "regression_results.json"
    with open(reg_path, "w") as f:
        json.dump(regression_results, f, indent=2, default=str)
    logger.info("Saved regression results to %s", reg_path)

    # 3. Semantic entropy
    print("\n=== Phase 3: Semantic Entropy ===")
    entropy_df = compute_semantic_entropy(df)
    if len(entropy_df) > 0:
        correlations = analyze_entropy_correlations(entropy_df)
        entropy_results = {
            "entropy_summary": {
                "n_pairs": len(entropy_df),
                "mean_entropy": float(entropy_df["semantic_entropy"].mean()),
                "std_entropy": float(entropy_df["semantic_entropy"].std()),
            },
            "correlations": correlations,
        }
        entropy_path = args.output_dir / "semantic_entropy.json"
        with open(entropy_path, "w") as f:
            json.dump(entropy_results, f, indent=2, default=str)
        entropy_df.to_parquet(args.output_dir / "semantic_entropy_detail.parquet", index=False)
        logger.info("Saved semantic entropy to %s", entropy_path)
    else:
        logger.warning("No repeated samples found; skipping semantic entropy")

    print("\n=== Analysis Complete ===")
    print(f"  All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
