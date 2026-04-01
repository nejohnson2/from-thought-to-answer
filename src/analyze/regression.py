"""Mixed-effects regression models for the CoT uncertainty study.

Fits the four regression models specified in the research design:
  Model A: Correctness
  Model B: Appropriate behavior
  Model C: Suppression
  Temporal: Calibration error ~ uncertainty position

Usage:
    python -m src.analyze.regression --input data/processed/all_responses.parquet \
        --output data/processed/regression_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for regression analysis."""
    # Filter to primary samples only
    primary = df[df["sample_index"] == 0].copy()

    # Ensure binary columns are numeric
    for col in [
        "uncertainty_in_artifact", "uncertainty_in_answer",
        "transmission", "suppression", "artifact_has_revision_event",
        "truncated", "parse_failure",
    ]:
        if col in primary.columns:
            primary[col] = primary[col].astype(int)

    # Create is_correct column (requires ground_truth comparison — placeholder)
    if "is_correct" not in primary.columns:
        primary["is_correct"] = np.nan
        logger.warning("'is_correct' column not found; correctness model will be skipped")

    # Create appropriate_behavior column
    if "is_correct" in primary.columns and "parsed_abstain" in primary.columns:
        primary["appropriate_behavior"] = (
            (primary["is_answerable"] & (primary["is_correct"] == 1))
            | (~primary["is_answerable"] & primary["parsed_abstain"])
        ).astype(int)

    # Create calibration_error (|confidence - accuracy|)
    if "parsed_confidence" in primary.columns and "is_correct" in primary.columns:
        primary["calibration_error"] = np.abs(
            primary["parsed_confidence"] / 100.0 - primary["is_correct"]
        )

    # Categorical columns for fixed effects
    primary["model"] = pd.Categorical(primary["model"])
    primary["task_bucket"] = pd.Categorical(primary["task_bucket"])
    primary["transparency_regime"] = pd.Categorical(primary["transparency_regime"])

    # Drop rows with failures
    primary = primary[~primary["parse_failure"].astype(bool)].copy()
    primary = primary[~primary["truncated"].astype(bool)].copy()

    return primary


def fit_model_a(df: pd.DataFrame) -> dict | None:
    """Model A: Correctness prediction.

    correct ~ uncertainty_in_artifact + uncertainty_in_answer + self_confidence +
              task_bucket + model + (1|prompt_id)
    """
    if df["is_correct"].isna().all():
        logger.warning("Skipping Model A: no correctness data")
        return None

    formula = (
        "is_correct ~ uncertainty_in_artifact + uncertainty_in_answer + "
        "parsed_confidence + C(task_bucket) + C(model)"
    )

    try:
        model = smf.mixedlm(
            formula,
            data=df.dropna(subset=["is_correct"]),
            groups="prompt_id",
        )
        result = model.fit(reml=False)
        return _extract_results(result, "Model A: Correctness")
    except Exception as e:
        logger.error("Model A failed: %s", e)
        return {"error": str(e)}


def fit_model_b(df: pd.DataFrame) -> dict | None:
    """Model B: Appropriate behavior prediction.

    appropriate ~ uncertainty_in_artifact + has_revision + task_bucket +
                 transparency_regime + (1|prompt_id)
    """
    if "appropriate_behavior" not in df.columns or df["appropriate_behavior"].isna().all():
        logger.warning("Skipping Model B: no appropriate_behavior data")
        return None

    formula = (
        "appropriate_behavior ~ uncertainty_in_artifact + artifact_has_revision_event + "
        "C(task_bucket) + C(transparency_regime)"
    )

    try:
        model = smf.mixedlm(
            formula,
            data=df.dropna(subset=["appropriate_behavior"]),
            groups="prompt_id",
        )
        result = model.fit(reml=False)
        return _extract_results(result, "Model B: Appropriate Behavior")
    except Exception as e:
        logger.error("Model B failed: %s", e)
        return {"error": str(e)}


def fit_model_c(df: pd.DataFrame) -> dict | None:
    """Model C: Suppression prediction.

    suppression ~ model + task_bucket + uncertainty_rate_artifact +
                 transparency_regime + (1|prompt_id)
    """
    # Only fit on responses with uncertainty in artifact
    df_with_uncertainty = df[df["uncertainty_in_artifact"] == 1].copy()

    if len(df_with_uncertainty) < 20:
        logger.warning("Skipping Model C: insufficient data with artifact uncertainty")
        return None

    formula = (
        "suppression ~ C(model) + C(task_bucket) + artifact_uncertainty_rate + "
        "C(transparency_regime)"
    )

    try:
        model = smf.mixedlm(
            formula,
            data=df_with_uncertainty,
            groups="prompt_id",
        )
        result = model.fit(reml=False)
        return _extract_results(result, "Model C: Suppression")
    except Exception as e:
        logger.error("Model C failed: %s", e)
        return {"error": str(e)}


def fit_temporal_model(df: pd.DataFrame) -> dict | None:
    """Temporal model: Calibration error ~ uncertainty position.

    calibration_error ~ first_uncertainty_pos * task_bucket + has_revision +
                       model + (1|prompt_id)
    """
    if "calibration_error" not in df.columns or df["calibration_error"].isna().all():
        logger.warning("Skipping temporal model: no calibration_error data")
        return None

    # Only fit on responses with uncertainty in artifact
    df_temporal = df[
        (df["uncertainty_in_artifact"] == 1)
        & (df["artifact_first_uncertainty_position"] >= 0)
    ].copy()

    if len(df_temporal) < 20:
        logger.warning("Skipping temporal model: insufficient data")
        return None

    formula = (
        "calibration_error ~ artifact_first_uncertainty_position * C(task_bucket) + "
        "artifact_has_revision_event + C(model)"
    )

    try:
        model = smf.mixedlm(
            formula,
            data=df_temporal.dropna(subset=["calibration_error"]),
            groups="prompt_id",
        )
        result = model.fit(reml=False)
        return _extract_results(result, "Temporal: Calibration Error")
    except Exception as e:
        logger.error("Temporal model failed: %s", e)
        return {"error": str(e)}


def _extract_results(result, model_name: str) -> dict:
    """Extract key results from a statsmodels MixedLM result."""
    summary = {
        "model_name": model_name,
        "n_observations": int(result.nobs),
        "n_groups": int(result.k_fe + result.k_re),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "converged": result.converged,
        "coefficients": {},
    }

    # Extract coefficients with CIs
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()
    bse = result.bse

    for name in params.index:
        coef = {
            "estimate": float(params[name]),
            "std_error": float(bse[name]),
            "p_value": float(pvalues[name]),
            "ci_lower": float(conf_int.loc[name, 0]),
            "ci_upper": float(conf_int.loc[name, 1]),
        }
        summary["coefficients"][name] = coef

    return summary


def apply_fdr_correction(results: dict) -> dict:
    """Apply Benjamini-Hochberg FDR correction across all p-values."""
    all_pvalues = []
    pvalue_keys = []

    for model_name, model_result in results.items():
        if model_result is None or "error" in model_result:
            continue
        for coef_name, coef_data in model_result.get("coefficients", {}).items():
            all_pvalues.append(coef_data["p_value"])
            pvalue_keys.append((model_name, coef_name))

    if not all_pvalues:
        return results

    # BH correction
    from statsmodels.stats.multitest import multipletests
    rejected, corrected, _, _ = multipletests(all_pvalues, method="fdr_bh")

    for (model_name, coef_name), p_corrected, is_sig in zip(pvalue_keys, corrected, rejected):
        results[model_name]["coefficients"][coef_name]["p_value_fdr"] = float(p_corrected)
        results[model_name]["coefficients"][coef_name]["significant_fdr"] = bool(is_sig)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit regression models")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/regression_results.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    prepared = prepare_data(df)
    logger.info("Prepared %d rows for analysis", len(prepared))

    results = {
        "model_a_correctness": fit_model_a(prepared),
        "model_b_appropriate": fit_model_b(prepared),
        "model_c_suppression": fit_model_c(prepared),
        "temporal_calibration": fit_temporal_model(prepared),
    }

    results = apply_fdr_correction(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved regression results to %s", args.output)

    # Print summary
    print("\n=== Regression Results ===")
    for name, result in results.items():
        if result is None:
            print(f"  {name}: SKIPPED")
        elif "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
        else:
            n_sig = sum(
                1 for c in result["coefficients"].values()
                if c.get("significant_fdr", c["p_value"] < 0.05)
            )
            print(f"  {name}: N={result['n_observations']}, AIC={result['aic']:.1f}, {n_sig} significant coefficients")


if __name__ == "__main__":
    main()
