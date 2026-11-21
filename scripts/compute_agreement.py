"""Compute inter-annotator agreement from two or more annotation CSVs.

Usage:
  python scripts/compute_agreement.py annotation/annotator_A.csv annotation/annotator_B.csv
"""
from __future__ import annotations

import argparse
import itertools
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="CSV files with columns: id, annotator_label")
    args = p.parse_args()
    dfs = []
    for i, f in enumerate(args.files):
        df = pd.read_csv(f)[["id", "annotator_label"]].copy()
        df = df.rename(columns={"annotator_label": f"label_{i}"})
        dfs.append(df)
    m = dfs[0]
    for df in dfs[1:]:
        m = m.merge(df, on="id", how="inner")
    label_cols = [c for c in m.columns if c.startswith("label_")]
    print(f"items={len(m)} annotators={len(label_cols)}")
    for a, b in itertools.combinations(label_cols, 2):
        agree = (m[a] == m[b]).mean()
        kappa = cohen_kappa_score(m[a], m[b])
        print(f"{a} vs {b}: percent_agreement={agree:.4f} cohen_kappa={kappa:.4f}")
    if len(label_cols) > 2:
        # simple mean pairwise kappa
        kappas = []
        for a, b in itertools.combinations(label_cols, 2):
            kappas.append(cohen_kappa_score(m[a], m[b]))
        print(f"mean_pairwise_cohen_kappa={sum(kappas)/len(kappas):.4f}")
    disagree = m[m[label_cols].nunique(axis=1) > 1]
    if not disagree.empty:
        disagree.to_csv("annotation/disagreements.csv", index=False)
        print(f"disagreements={len(disagree)} saved=annotation/disagreements.csv")


if __name__ == "__main__":
    main()