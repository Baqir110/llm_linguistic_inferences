"""Evaluation metrics, baselines, significance tests and linguistic error attribution."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

LABELS = ["Yes", "No"]
CATEGORY_KEYS = {
    "lexical_entailment": "lexical",
    "presupposition": "presupposition",
    "plausible_non_entailment": "plausible_non_ent",
}


def classify_error(row: pd.Series) -> str:
    """Heuristic attribution of each prediction to a failure mode (uses subtype names)."""
    gold, pred = row["gold_label"], row["predicted_label"]
    cat, sub = row["category"], row["subtype"]

    if pred not in LABELS:
        return "format_error"
    if pred == gold:
        return "correct"

    if gold == "No" and pred == "Yes":  # over-entailment
        if cat == "plausible_non_entailment":
            return "world_knowledge_overentailment"
        if any(k in sub for k in ("non_factive", "counterfactive", "suppositional", "implicative_failure",
                                  "counterfactual", "almost")):
            return "factivity_overgeneralization"
        if "negated" in sub or "filter" in sub or "cancellation" in sub:
            return "negation_scope_failure"
        if "reverse" in sub or "privative" in sub or "non_intersective" in sub:
            return "upward_monotonicity_bias"
        if "co_hyponym" in sub:
            return "sibling_term_confusion"
        return "false_positive_other"

    if gold == "Yes" and pred == "No":  # under-entailment
        if cat == "plausible_non_entailment":
            return "overly_strict_on_control"
        if "negated" in sub or "downward" in sub or sub == "question_projection":
            return "negation_scope_failure"
        if cat == "presupposition":
            return "presupposition_projection_failure"
        if cat == "lexical_entailment":
            return "lexical_entailment_failure"
        return "false_negative_other"
    return "unclassified"


def bootstrap_accuracy_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of accuracy over items."""
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _macro_f1(gold, pred) -> float:
    # 'Invalid' predictions count as errors (they are in neither label's predicted set).
    return float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0))


def compute_summary_metrics(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Per (model, strategy): accuracy (+CI), macro-F1, balanced accuracy, FP/FN, invalid rate,
    and per-category accuracy, macro-F1 and balanced accuracy."""
    records = []
    for (model, strategy), g in df.groupby(["model", "strategy"]):
        correct = (g["gold_label"] == g["predicted_label"]).to_numpy()
        lo, hi = bootstrap_accuracy_ci(correct, n_boot=n_boot)
        neg, pos = g[g["gold_label"] == "No"], g[g["gold_label"] == "Yes"]
        rec = {
            "model": model,
            "strategy": strategy,
            "n_samples": len(g),
            "accuracy": round(float(correct.mean()), 4),
            "acc_ci_low": round(lo, 4),
            "acc_ci_high": round(hi, 4),
            "macro_f1": round(_macro_f1(g["gold_label"], g["predicted_label"]), 4),
            "balanced_accuracy": round(
                float(np.nanmean([(pos["predicted_label"] == "Yes").mean() if len(pos) else np.nan,
                                  (neg["predicted_label"] == "No").mean() if len(neg) else np.nan])), 4),
            "false_positive_rate": round(float((neg["predicted_label"] == "Yes").mean()) if len(neg) else 0.0, 4),
            "false_negative_rate": round(float((pos["predicted_label"] == "No").mean()) if len(pos) else 0.0, 4),
            "invalid_rate": round(float((~g["predicted_label"].isin(LABELS)).mean()), 4),
        }
        if "contested" in g.columns:  # accuracy without the items whose gold label is debatable
            contested = g["contested"].map(lambda v: v is True or str(v).strip().lower() == "true")
            u = g[~contested]
            rec["acc_excl_contested"] = round(float((u["gold_label"] == u["predicted_label"]).mean()), 4) if len(u) else np.nan
        for cat, key in CATEGORY_KEYS.items():
            c = g[g["category"] == cat]
            rec[f"acc_{key}"] = round(float((c["gold_label"] == c["predicted_label"]).mean()), 4) if len(c) else np.nan
            rec[f"f1_{key}"] = round(_macro_f1(c["gold_label"], c["predicted_label"]), 4) if len(c) else np.nan
            rec[f"bacc_{key}"] = (
                round(float(np.nanmean([
                    (c[c.gold_label == "Yes"].predicted_label == "Yes").mean() if (c.gold_label == "Yes").any() else np.nan,
                    (c[c.gold_label == "No"].predicted_label == "No").mean() if (c.gold_label == "No").any() else np.nan,
                ])), 4) if len(c) else np.nan
            )
        records.append(rec)
    return pd.DataFrame(records)


def majority_baselines(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of a constant 'always Yes' / 'always No' guesser, overall and per category.
    Needed because the label distribution is skewed within categories."""
    rows = []
    for name, sub in [("overall", benchmark)] + [(c, benchmark[benchmark.category == c]) for c in CATEGORY_KEYS]:
        rows.append({
            "scope": name, "n": len(sub),
            "always_yes_acc": round(float((sub.gold_label == "Yes").mean()), 4),
            "always_no_acc": round(float((sub.gold_label == "No").mean()), 4),
        })
    return pd.DataFrame(rows)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired per-item correctness. Returns (b, c, p)."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_significance(df: pd.DataFrame) -> pd.DataFrame:
    """McNemar tests: (i) strategies within each model vs zero_shot, (ii) models within each strategy."""
    d = df.assign(correct=df["gold_label"] == df["predicted_label"])
    wide = d.pivot_table(index="id", columns=["model", "strategy"], values="correct", aggfunc="first")
    rows = []

    def test(col_a, col_b, kind):
        sub = wide[[col_a, col_b]].dropna()
        if sub.empty:
            return
        b, c, p = mcnemar_exact(sub[col_a].to_numpy(dtype=bool), sub[col_b].to_numpy(dtype=bool))
        rows.append({"comparison": kind, "a": f"{col_a[0]}|{col_a[1]}", "b": f"{col_b[0]}|{col_b[1]}",
                     "a_only_correct": b, "b_only_correct": c, "p_value": round(p, 4)})

    models = sorted({m for m, _ in wide.columns})
    strategies = sorted({s for _, s in wide.columns})
    for m in models:
        if (m, "zero_shot") in wide.columns:
            for s in strategies:
                if s != "zero_shot" and (m, s) in wide.columns:
                    test((m, "zero_shot"), (m, s), "strategy")
    for s in strategies:
        for m1, m2 in itertools.combinations(models, 2):
            if (m1, s) in wide.columns and (m2, s) in wide.columns:
                test((m1, s), (m2, s), "model")
    return pd.DataFrame(rows)


def compute_perturbation_consistency(base_df: pd.DataFrame, pert_df: pd.DataFrame) -> pd.DataFrame:
    """Label stability across truth-preserving paraphrases of the same item."""
    merged = pd.merge(
        base_df[["model", "strategy", "id", "gold_label", "predicted_label"]],
        pert_df[["model", "strategy", "base_id", "predicted_label"]],
        left_on=["model", "strategy", "id"],
        right_on=["model", "strategy", "base_id"],
        suffixes=("_orig", "_pert"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent"] = merged["predicted_label_orig"] == merged["predicted_label_pert"]
    merged["orig_correct"] = merged["predicted_label_orig"] == merged["gold_label"]
    merged["pert_correct"] = merged["predicted_label_pert"] == merged["gold_label"]
    merged["both_correct"] = merged["orig_correct"] & merged["pert_correct"]
    return (
        merged.groupby(["model", "strategy"])
        .agg(pairs=("id", "count"), consistency_rate=("consistent", "mean"),
             joint_accuracy=("both_correct", "mean"),
             acc_original=("orig_correct", "mean"), acc_perturbed=("pert_correct", "mean"))
        .reset_index()
        .round(4)
    )
