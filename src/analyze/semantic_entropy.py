"""Semantic entropy computation from repeated sampling.

Clusters semantically equivalent answers and computes entropy over the
cluster distribution. Used to compare behavioral uncertainty (from sampling)
with expressed uncertainty (from artifact markers).

Usage:
    python -m src.analyze.semantic_entropy --input data/processed/all_responses.parquet \
        --output data/processed/semantic_entropy.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, lowercases, removes common prefixes/suffixes,
    and normalizes numeric representations.
    """
    text = answer.strip().lower()

    # Remove common prefixes
    for prefix in ["the answer is ", "answer: ", "final answer: "]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    # Strip trailing punctuation
    text = text.rstrip(".")

    # Try to normalize numbers
    try:
        num = float(text.replace(",", "").replace("$", "").replace("%", ""))
        # Use a consistent format for numbers
        if num == int(num):
            return str(int(num))
        return f"{num:.6g}"
    except ValueError:
        pass

    return text.strip()


def cluster_answers(answers: list[str]) -> list[list[int]]:
    """Cluster answers by exact normalized match.

    For a more sophisticated approach, use an embedding model to compute
    semantic similarity. This simple version uses exact string matching
    after normalization, which works well for math and factual QA.

    Args:
        answers: List of answer strings.

    Returns:
        List of clusters, where each cluster is a list of indices.
    """
    normalized = [normalize_answer(a) for a in answers]
    clusters: dict[str, list[int]] = {}

    for idx, norm in enumerate(normalized):
        if norm in clusters:
            clusters[norm].append(idx)
        else:
            clusters[norm] = [idx]

    return list(clusters.values())


def compute_entropy(clusters: list[list[int]], total: int) -> float:
    """Compute entropy over cluster distribution.

    Args:
        clusters: List of clusters (each a list of sample indices).
        total: Total number of samples.

    Returns:
        Shannon entropy in nats.
    """
    if total == 0:
        return 0.0

    probs = np.array([len(c) / total for c in clusters])
    # Filter zero probabilities
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log(probs)))


def compute_semantic_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute semantic entropy for all prompts with repeated samples.

    Args:
        df: Full annotated DataFrame including repeated samples.

    Returns:
        DataFrame with one row per (prompt_id, model) with entropy stats.
    """
    # Filter to prompts that have repeated samples
    repeated = df[df["sample_index"] > 0].copy()
    if len(repeated) == 0:
        logger.warning("No repeated samples found in data")
        return pd.DataFrame()

    # Get unique (prompt_id, model) combinations with repeated samples
    prompt_model_pairs = repeated.groupby(["prompt_id", "model"]).size().reset_index(name="n_samples")
    logger.info("Found %d (prompt, model) pairs with repeated samples", len(prompt_model_pairs))

    results = []
    for _, row in prompt_model_pairs.iterrows():
        pid = row["prompt_id"]
        model = row["model"]

        # Get all samples for this prompt-model pair (including primary)
        samples = df[(df["prompt_id"] == pid) & (df["model"] == model)].copy()
        answers = samples["parsed_final_answer"].tolist()

        if len(answers) < 2:
            continue

        clusters = cluster_answers(answers)
        entropy = compute_entropy(clusters, len(answers))

        # Also compute answer dispersion (number of unique clusters)
        n_unique = len(clusters)

        # Get primary sample's artifact features for comparison
        primary = samples[samples["sample_index"] == 0]
        artifact_uncertainty_rate = (
            float(primary["artifact_uncertainty_rate"].iloc[0])
            if len(primary) > 0 and "artifact_uncertainty_rate" in primary.columns
            else None
        )
        has_uncertainty_in_artifact = (
            bool(primary["uncertainty_in_artifact"].iloc[0])
            if len(primary) > 0
            else None
        )

        results.append({
            "prompt_id": pid,
            "model": model,
            "n_samples": len(answers),
            "n_unique_answers": n_unique,
            "semantic_entropy": entropy,
            "max_cluster_size": max(len(c) for c in clusters),
            "artifact_uncertainty_rate": artifact_uncertainty_rate,
            "uncertainty_in_artifact": has_uncertainty_in_artifact,
        })

    return pd.DataFrame(results)


def analyze_entropy_correlations(entropy_df: pd.DataFrame) -> dict:
    """Analyze correlations between semantic entropy and artifact features."""
    if len(entropy_df) < 5:
        return {"error": "insufficient data"}

    from scipy.stats import pearsonr, spearmanr

    results = {}

    # Correlation: semantic entropy vs artifact uncertainty rate
    valid = entropy_df.dropna(subset=["artifact_uncertainty_rate"])
    if len(valid) >= 5:
        r, p = pearsonr(valid["semantic_entropy"], valid["artifact_uncertainty_rate"])
        rho, rho_p = spearmanr(valid["semantic_entropy"], valid["artifact_uncertainty_rate"])
        results["entropy_vs_artifact_rate"] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "n": len(valid),
        }

    # Point-biserial: semantic entropy vs binary artifact uncertainty
    valid_bin = entropy_df.dropna(subset=["uncertainty_in_artifact"])
    if len(valid_bin) >= 5:
        groups = valid_bin.groupby("uncertainty_in_artifact")["semantic_entropy"]
        if len(groups) == 2:
            g_true = groups.get_group(True) if True in groups.groups else pd.Series()
            g_false = groups.get_group(False) if False in groups.groups else pd.Series()
            if len(g_true) >= 2 and len(g_false) >= 2:
                from scipy.stats import mannwhitneyu
                u, p = mannwhitneyu(g_true, g_false, alternative="greater")
                results["entropy_by_artifact_uncertainty"] = {
                    "mean_with_uncertainty": float(g_true.mean()),
                    "mean_without_uncertainty": float(g_false.mean()),
                    "mann_whitney_u": float(u),
                    "p_value": float(p),
                    "n_with": len(g_true),
                    "n_without": len(g_false),
                }

    # Per-model breakdown
    results["by_model"] = {}
    for model, group in entropy_df.groupby("model"):
        results["by_model"][str(model)] = {
            "mean_entropy": float(group["semantic_entropy"].mean()),
            "std_entropy": float(group["semantic_entropy"].std()),
            "mean_unique_answers": float(group["n_unique_answers"].mean()),
            "n": len(group),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute semantic entropy from repeated samples")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/semantic_entropy.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    entropy_df = compute_semantic_entropy(df)
    if len(entropy_df) == 0:
        logger.error("No semantic entropy data computed")
        sys.exit(1)

    logger.info("Computed entropy for %d (prompt, model) pairs", len(entropy_df))

    correlations = analyze_entropy_correlations(entropy_df)

    results = {
        "entropy_summary": {
            "n_pairs": len(entropy_df),
            "mean_entropy": float(entropy_df["semantic_entropy"].mean()),
            "std_entropy": float(entropy_df["semantic_entropy"].std()),
            "mean_unique_answers": float(entropy_df["n_unique_answers"].mean()),
        },
        "correlations": correlations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save entropy DataFrame
    entropy_path = args.output.parent / "semantic_entropy_detail.parquet"
    entropy_df.to_parquet(entropy_path, index=False)

    logger.info("Saved results to %s", args.output)

    print("\n=== Semantic Entropy Summary ===")
    print(f"  Pairs analyzed: {len(entropy_df)}")
    print(f"  Mean entropy: {entropy_df['semantic_entropy'].mean():.3f}")
    print(f"  Mean unique answers: {entropy_df['n_unique_answers'].mean():.1f}")
    if "entropy_vs_artifact_rate" in correlations:
        c = correlations["entropy_vs_artifact_rate"]
        print(f"  Entropy vs artifact rate: r={c['pearson_r']:.3f}, p={c['pearson_p']:.4f}")


if __name__ == "__main__":
    main()
