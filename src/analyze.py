"""Aggregate results across models: tables, baselines, significance tests, figures.

Run: python -m src.analyze
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import (compute_perturbation_consistency, compute_summary_metrics,
                         majority_baselines, paired_significance)

CATS = [("acc_lexical", "Lexical"), ("acc_presupposition", "Presupposition"),
        ("acc_plausible_non_ent", "Plausible non-ent.")]


def _read_all(res_path: Path, pattern: str) -> pd.DataFrame:
    files = sorted(f for f in glob.glob(str(res_path / pattern)) if "_limit" not in f)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def aggregate_and_plot(results_dir: str = "results", data: str = "data/inference_benchmark.csv") -> None:
    res = Path(results_dir)
    preds = _read_all(res, "predictions_*.csv")
    if preds.empty:
        raise FileNotFoundError(f"No prediction CSV files found in {results_dir}")
    preds = preds.drop_duplicates(subset=["model", "strategy", "id"], keep="last")
    preds.to_csv(res / "all_predictions.csv", index=False)

    summary = compute_summary_metrics(preds)
    summary.to_csv(res / "master_summary.csv", index=False)

    primary_cols = [c for c in [
        "model", "strategy", "n_samples", "accuracy", "acc_ci_low", "acc_ci_high",
        "macro_f1", "balanced_accuracy",
        "acc_lexical", "f1_lexical", "bacc_lexical",
        "acc_presupposition", "f1_presupposition", "bacc_presupposition",
        "acc_plausible_non_ent", "f1_plausible_non_ent", "bacc_plausible_non_ent",
    ] if c in summary.columns]
    exploratory_cols = [c for c in [
        "model", "strategy", "false_positive_rate", "false_negative_rate", "invalid_rate", "acc_excl_contested",
    ] if c in summary.columns]
    summary[primary_cols].to_csv(res / "primary_summary.csv", index=False)
    summary[exploratory_cols].to_csv(res / "exploratory_summary.csv", index=False)
    (res / "metrics_designation.json").write_text(json.dumps({
        "primary": primary_cols,
        "exploratory": exploratory_cols + ["error_attribution", "robustness_consistency", "logprob_margin"],
        "note": "Primary metrics are pre-registered for main comparisons; exploratory metrics are diagnostic and should be interpreted cautiously."
    }, indent=2), encoding="utf-8")

    majority_baselines(pd.read_csv(data)).to_csv(res / "majority_baselines.csv", index=False)
    paired_significance(preds).to_csv(res / "significance_tests.csv", index=False)

    if "error_type" in preds.columns:
        errors = preds[preds["error_type"] != "correct"]
        (errors.groupby(["model", "strategy", "error_type"]).size().unstack(fill_value=0)
         .reset_index().to_csv(res / "error_attribution.csv", index=False))

    pert = _read_all(res, "perturbation_predictions_*.csv")
    if not pert.empty:
        compute_perturbation_consistency(preds, pert).to_csv(res / "robustness_summary.csv", index=False)

    strategies = [s for s in ["zero_shot", "few_shot", "cot"] if s in set(summary.strategy)]
    # Fig 1: category accuracy per strategy
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.5 * len(strategies), 4.8), sharey=True, squeeze=False)
    for ax, strat in zip(axes[0], strategies):
        sub = summary[summary.strategy == strat].set_index("model")
        sub[[c for c, _ in CATS]].plot(kind="bar", ax=ax, rot=20, width=0.75)
        ax.set_title(strat.replace("_", "-"))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend([n for _, n in CATS], fontsize=8)
    axes[0][0].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(res / "category_accuracy_comparison.png", dpi=300)
    plt.close(fig)

    # Fig 2: over- vs under-entailment (FP vs FN rate)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (summary.assign(label=summary.model + " | " + summary.strategy).set_index("label")
     [["false_positive_rate", "false_negative_rate"]].plot(kind="bar", ax=ax, rot=45))
    ax.set_ylabel("Rate")
    ax.legend(["False positive (over-entailment)", "False negative (under-entailment)"])
    plt.tight_layout()
    plt.savefig(res / "fp_fn_rates.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    aggregate_and_plot()
