#!/usr/bin/env python3
"""Compare Ollama results across prompt templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prompt-sweep metrics.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--baseline-prompt", type=str, default="neutral")
    return parser.parse_args()


def read_prompt_metrics(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(results_root.glob("*/*/metrics.csv")):
        prompt = metrics_path.parent.parent.name
        dataset = metrics_path.parent.name
        metrics = pd.read_csv(metrics_path)
        if len(metrics) == 0:
            continue
        row = metrics.iloc[0].to_dict()
        config_path = metrics_path.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            row["model"] = config.get("model", row.get("model"))
            row["prompt_template"] = config.get("prompt_template", prompt)
            row["prompt_template_description"] = config.get("prompt_template_description", "")
        else:
            row["prompt_template"] = prompt
        row["dataset"] = dataset
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def add_baseline_deltas(frame: pd.DataFrame, baseline_prompt: str) -> pd.DataFrame:
    if len(frame) == 0:
        return frame
    baseline = frame[frame["prompt_template"] == baseline_prompt][
        ["dataset", "accuracy", "f1", "precision", "recall", "predicted_positive_rate"]
    ].rename(
        columns={
            "accuracy": "baseline_accuracy",
            "f1": "baseline_f1",
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "predicted_positive_rate": "baseline_predicted_positive_rate",
        }
    )
    out = frame.merge(baseline, on="dataset", how="left")
    for metric in ["accuracy", "f1", "precision", "recall", "predicted_positive_rate"]:
        out[f"{metric}_delta_vs_{baseline_prompt}"] = (
            out[metric] - out[f"baseline_{metric}"]
        )
    return out


def aggregate_by_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        return frame
    metrics = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "valid_prediction_rate",
        "predicted_positive_rate",
        "mean_batch_elapsed_s",
    ]
    available = [metric for metric in metrics if metric in frame.columns]
    grouped = frame.groupby("prompt_template", as_index=False)[available].mean()
    grouped = grouped.sort_values(["accuracy", "f1"], ascending=[False, False])
    return grouped


def main() -> None:
    args = parse_args()
    frame = read_prompt_metrics(args.results_root)
    frame = add_baseline_deltas(frame, args.baseline_prompt)

    preferred = [
        "dataset",
        "model",
        "prompt_template",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "valid_prediction_rate",
        "predicted_positive_rate",
        f"accuracy_delta_vs_{args.baseline_prompt}",
        f"f1_delta_vs_{args.baseline_prompt}",
        f"predicted_positive_rate_delta_vs_{args.baseline_prompt}",
        "mean_batch_elapsed_s",
        "total_valid_predictions",
        "total_expected_predictions",
        "metrics_path",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + remaining] if len(frame) else frame

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")

    aggregate = aggregate_by_prompt(frame)
    if args.aggregate_csv is not None:
        args.aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(args.aggregate_csv, index=False)
        print(f"Wrote {args.aggregate_csv}")

    if len(aggregate):
        print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
