"""Publication-ready figure generation for the CoT uncertainty study.

Reads saved analysis results and generates figures. Completely separated
from the analytical pipeline — figures can be regenerated without re-running
analysis.

Usage:
    python -m src.visualize.figures --stats-dir data/processed \
        --output-dir paper/neurips_2026/figures
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette: colorblind-friendly
MODEL_COLORS = {
    "deepseek-r1:70b": "#0072B2",
    "qwen3:32b": "#E69F00",
    "claude-sonnet-4-6-20260401": "#CC79A7",
    "gemini-2.5-flash": "#009E73",
    "o4-mini": "#D55E00",
}

REGIME_COLORS = {
    "raw_trace": "#0072B2",
    "summarized_artifact": "#D55E00",
}


def fig1_suppression_heatmap(stats: dict, output_dir: Path) -> None:
    """Figure 1: Suppression rate heatmap (model x task bucket)."""
    cross_stats = stats.get("by_model_and_bucket", {})
    if not cross_stats:
        logger.warning("No cross stats for heatmap")
        return

    models = sorted(set(k.split("|")[0] for k in cross_stats.keys()))
    buckets = sorted(set(k.split("|")[1] for k in cross_stats.keys()))

    data = np.full((len(models), len(buckets)), np.nan)
    for i, model in enumerate(models):
        for j, bucket in enumerate(buckets):
            key = f"{model}|{bucket}"
            if key in cross_stats:
                data[i, j] = cross_stats[key].get("suppression_rate", 0) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([b.replace("_", "\n") for b in buckets], rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("/")[-1] for m in models])

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(buckets)):
            if not np.isnan(data[i, j]):
                color = "white" if data[i, j] > 50 else "black"
                ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center", color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Suppression Rate (%)")
    ax.set_title("Uncertainty Suppression: Artifact → Answer")

    fig.savefig(output_dir / "fig1_suppression_heatmap.pdf")
    fig.savefig(output_dir / "fig1_suppression_heatmap.png")
    plt.close(fig)
    logger.info("Saved fig1_suppression_heatmap")


def fig2_transmission_flow(df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 2: Transmission/suppression rates by model (stacked bar)."""
    primary = df[df["sample_index"] == 0].copy()

    # Compute rates per model
    model_stats = []
    for model, group in primary.groupby("model"):
        n = len(group)
        has_artifact_unc = group["uncertainty_in_artifact"].sum()
        if has_artifact_unc == 0:
            continue
        unc_subset = group[group["uncertainty_in_artifact"]]
        transmitted = unc_subset["transmission"].sum() / has_artifact_unc * 100
        suppressed = unc_subset["suppression"].sum() / has_artifact_unc * 100
        model_stats.append({
            "model": model.split("/")[-1],
            "regime": group["transparency_regime"].iloc[0],
            "transmitted": transmitted,
            "suppressed": suppressed,
        })

    if not model_stats:
        logger.warning("No data for transmission flow figure")
        return

    stats_df = pd.DataFrame(model_stats)
    stats_df = stats_df.sort_values("regime")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(stats_df))

    bars1 = ax.bar(x, stats_df["transmitted"], label="Transmitted", color="#009E73", alpha=0.85)
    bars2 = ax.bar(x, stats_df["suppressed"], bottom=stats_df["transmitted"],
                   label="Suppressed", color="#D55E00", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["model"], rotation=30, ha="right")
    ax.set_ylabel("% of Responses with Artifact Uncertainty")
    ax.set_title("Uncertainty Transmission vs. Suppression")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    # Add regime annotation
    for i, (_, row) in enumerate(stats_df.iterrows()):
        regime_label = "R" if row["regime"] == "raw_trace" else "S"
        ax.text(i, -5, regime_label, ha="center", va="top", fontsize=8,
                color=REGIME_COLORS.get(row["regime"], "gray"), fontweight="bold")

    fig.savefig(output_dir / "fig2_transmission_flow.pdf")
    fig.savefig(output_dir / "fig2_transmission_flow.png")
    plt.close(fig)
    logger.info("Saved fig2_transmission_flow")


def fig3_uncertainty_position(df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 3: Distribution of first uncertainty position in reasoning artifacts."""
    primary = df[(df["sample_index"] == 0) & (df["uncertainty_in_artifact"])].copy()
    if len(primary) == 0:
        logger.warning("No data for position figure")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for model in primary["model"].unique():
        model_data = primary[primary["model"] == model]
        positions = model_data["artifact_first_uncertainty_position"]
        valid = positions[positions >= 0]
        if len(valid) < 5:
            continue
        color = MODEL_COLORS.get(model, None)
        sns.kdeplot(valid, ax=ax, label=model.split("/")[-1], color=color, linewidth=1.5)

    ax.set_xlabel("Normalized Position in Reasoning Artifact (0 = start, 1 = end)")
    ax.set_ylabel("Density")
    ax.set_title("Where Uncertainty First Appears in Reasoning")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1)

    fig.savefig(output_dir / "fig3_uncertainty_position.pdf")
    fig.savefig(output_dir / "fig3_uncertainty_position.png")
    plt.close(fig)
    logger.info("Saved fig3_uncertainty_position")


def fig4_regime_comparison(stats: dict, output_dir: Path) -> None:
    """Figure 4: Key metrics compared across transparency regimes."""
    regime_stats = stats.get("by_regime", {})
    if not regime_stats:
        logger.warning("No regime stats for comparison figure")
        return

    metrics = [
        ("uncertainty_in_artifact_rate", "Uncertainty\nin Artifact"),
        ("uncertainty_in_answer_rate", "Uncertainty\nin Answer"),
        ("transmission_rate", "Transmission"),
        ("suppression_rate", "Suppression"),
        ("revision_event_rate", "Revision\nEvents"),
    ]

    regimes = list(regime_stats.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, regime in enumerate(regimes):
        values = [regime_stats[regime].get(m[0], 0) * 100 for m in metrics]
        color = REGIME_COLORS.get(regime, "gray")
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=regime.replace("_", " ").title(),
                      color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Uncertainty Behavior by Transparency Regime")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    fig.savefig(output_dir / "fig4_regime_comparison.pdf")
    fig.savefig(output_dir / "fig4_regime_comparison.png")
    plt.close(fig)
    logger.info("Saved fig4_regime_comparison")


def fig5_entropy_correlation(entropy_path: Path, output_dir: Path) -> None:
    """Figure 5: Semantic entropy vs. artifact uncertainty rate (scatter)."""
    if not entropy_path.exists():
        logger.warning("No semantic entropy data for scatter plot")
        return

    entropy_df = pd.read_parquet(entropy_path)
    valid = entropy_df.dropna(subset=["artifact_uncertainty_rate"])
    if len(valid) < 10:
        logger.warning("Insufficient data for entropy scatter")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for model in valid["model"].unique():
        model_data = valid[valid["model"] == model]
        color = MODEL_COLORS.get(model, None)
        ax.scatter(
            model_data["artifact_uncertainty_rate"],
            model_data["semantic_entropy"],
            label=model.split("/")[-1],
            color=color,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Artifact Uncertainty Rate (markers per 100 tokens)")
    ax.set_ylabel("Semantic Entropy (nats)")
    ax.set_title("Expressed vs. Behavioral Uncertainty")
    ax.legend(loc="upper left", fontsize=8)

    # Add trend line
    from numpy.polynomial import polynomial as P
    x_all = valid["artifact_uncertainty_rate"].values
    y_all = valid["semantic_entropy"].values
    if len(x_all) > 5:
        coeffs = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.5, linewidth=1)

    fig.savefig(output_dir / "fig5_entropy_correlation.pdf")
    fig.savefig(output_dir / "fig5_entropy_correlation.png")
    plt.close(fig)
    logger.info("Saved fig5_entropy_correlation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--stats-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/neurips_2026/figures"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load descriptive stats
    desc_path = args.stats_dir / "descriptive_stats.json"
    if desc_path.exists():
        with open(desc_path) as f:
            desc_stats = json.load(f)
    else:
        logger.error("Descriptive stats not found at %s", desc_path)
        desc_stats = {}

    # Load annotated data for detailed figures
    data_path = args.stats_dir / "all_responses.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        logger.error("Annotated data not found at %s", data_path)
        df = pd.DataFrame()

    # Generate figures
    if desc_stats:
        fig1_suppression_heatmap(desc_stats, args.output_dir)
        fig4_regime_comparison(desc_stats, args.output_dir)

    if len(df) > 0:
        fig2_transmission_flow(df, args.output_dir)
        fig3_uncertainty_position(df, args.output_dir)

    entropy_path = args.stats_dir / "semantic_entropy_detail.parquet"
    fig5_entropy_correlation(entropy_path, args.output_dir)

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
