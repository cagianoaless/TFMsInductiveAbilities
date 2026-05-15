#!/usr/bin/env python3
"""Summarize Ollama LLM results and optionally compare them with TabPFN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM-vs-TabPFN summary table.")
    parser.add_argument("--llm-results-root", type=Path, default=Path("llm_exp/results"))
    parser.add_argument("--tabpfn-results-root", type=Path, default=Path("tfm_results_tabpfn"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("llm_exp/ollama_vs_tabpfn_summary.csv"),
    )
    return parser.parse_args()


def read_metrics(root: Path, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        frame = pd.read_csv(metrics_path)
        if len(frame) == 0:
            continue
        row = frame.iloc[0].to_dict()
        row["dataset"] = metrics_path.parent.name
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["dataset"])
    out = pd.DataFrame(rows)
    return out.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in out.columns
            if column not in {"dataset", "metrics_path"}
        }
    ).rename(columns={"metrics_path": f"{prefix}_metrics_path"})


def main() -> None:
    args = parse_args()
    llm = read_metrics(args.llm_results_root, "llm")
    tabpfn = read_metrics(args.tabpfn_results_root, "tabpfn")
    if len(tabpfn):
        summary = llm.merge(tabpfn, on="dataset", how="left")
    else:
        summary = llm

    preferred = [
        "dataset",
        "llm_model",
        "llm_accuracy",
        "llm_f1",
        "llm_precision",
        "llm_recall",
        "llm_valid_prediction_rate",
        "llm_total_valid_predictions",
        "llm_total_expected_predictions",
        "tabpfn_model",
        "tabpfn_accuracy",
        "tabpfn_f1",
        "tabpfn_precision",
        "tabpfn_recall",
        "tabpfn_average_precision",
        "tabpfn_roc_auc",
    ]
    ordered = [column for column in preferred if column in summary.columns]
    remaining = [column for column in summary.columns if column not in ordered]
    summary = summary[ordered + remaining]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    if len(summary):
        print(summary[ordered].to_string(index=False))


if __name__ == "__main__":
    main()
