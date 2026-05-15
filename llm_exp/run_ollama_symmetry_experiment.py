#!/usr/bin/env python3
"""Evaluate a local Ollama model on binary relation symmetry datasets.

The script samples several independent ICL/test batches instead of sending the
whole table in one prompt. This makes the protocol usable with local LLMs and
keeps the result comparable to the TabPFN runner: fit examples come from
``split == train`` and evaluated rows come from ``split == test``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PromptBatch:
    run_idx: int
    prompt: str
    icl_indices: list[int]
    test_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Ollama LLM on a symmetry-style binary relation dataset."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument(
        "--backend",
        choices=["local", "cloud"],
        default="local",
        help=(
            "local uses a local Ollama host, including local proxy access to signed-in cloud models. "
            "cloud calls https://ollama.com directly and requires an API key."
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=["chat", "generate"],
        default="generate",
        help="Ollama API endpoint. Ollama Cloud examples use chat; generate is kept for local compatibility.",
    )
    parser.add_argument(
        "--structured-output",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Whether to send format=json. auto enables it for normal local models and disables it for cloud models, "
            "because Ollama Cloud may not support structured outputs."
        ),
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--ollama-api-key-env",
        type=str,
        default="OLLAMA_API_KEY",
        help="Environment variable containing the Ollama Cloud API key when --backend cloud is used.",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--icl-size", type=int, default=64)
    parser.add_argument("--test-size-per-run", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--entity-1-column", type=str, default="entity_1")
    parser.add_argument("--entity-2-column", type=str, default="entity_2")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--split-column", type=str, default="split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--relation-name",
        type=str,
        default="R",
        help="Neutral relation name used in the prompt. Avoid names like friendship if you do not want to reveal symmetry.",
    )
    parser.add_argument(
        "--sampling",
        choices=["balanced", "natural"],
        default="balanced",
        help="Balanced samples equal numbers of positive/negative rows when possible.",
    )
    parser.add_argument(
        "--reveal-symmetry",
        action="store_true",
        help="Sanity-check mode: explicitly tells the LLM that the relation is symmetric.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save full prompts and raw responses to raw_responses.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, args: argparse.Namespace) -> None:
    required = [
        args.entity_1_column,
        args.entity_2_column,
        args.label_column,
        args.split_column,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
    if isinstance(value, float) and value in (0.0, 1.0) and not math.isnan(value):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "positive", "pos"}:
        return 1
    if text in {"0", "false", "no", "negative", "neg"}:
        return 0
    raise ValueError(f"Cannot normalize label value: {value!r}")


def sample_rows(
    frame: pd.DataFrame,
    size: int,
    rng: random.Random,
    label_column: str,
    sampling: str,
) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("Sample size must be positive.")
    if len(frame) == 0:
        raise ValueError("Cannot sample from an empty frame.")
    sample_size = min(size, len(frame))
    if sampling == "natural":
        return frame.sample(n=sample_size, random_state=rng.randrange(2**32))

    positives = frame[frame[label_column] == 1]
    negatives = frame[frame[label_column] == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return frame.sample(n=sample_size, random_state=rng.randrange(2**32))

    positive_count = min(len(positives), sample_size // 2)
    negative_count = min(len(negatives), sample_size - positive_count)
    remaining = sample_size - positive_count - negative_count
    if remaining > 0:
        if len(positives) - positive_count > len(negatives) - negative_count:
            positive_count += remaining
        else:
            negative_count += remaining

    pos_sample = positives.sample(n=positive_count, random_state=rng.randrange(2**32))
    neg_sample = negatives.sample(n=negative_count, random_state=rng.randrange(2**32))
    out = pd.concat([pos_sample, neg_sample], axis=0)
    return out.sample(frac=1.0, random_state=rng.randrange(2**32))


def format_pair(row: pd.Series, args: argparse.Namespace) -> str:
    left = str(row[args.entity_1_column])
    right = str(row[args.entity_2_column])
    return f"({left}, {right})"


def build_prompt(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    args: argparse.Namespace,
    run_idx: int,
) -> str:
    relation_hint = (
        f"The relation {args.relation_name} is symmetric: if R(a,b)=1 then R(b,a)=1, and if R(a,b)=0 then R(b,a)=0."
        if args.reveal_symmetry
        else f"You are given examples of an unknown binary relation {args.relation_name} between entity identifiers."
    )
    lines = [
        "Task: binary classification of ordered entity pairs.",
        relation_hint,
        "A label of 1 means the relation is true for the ordered pair. A label of 0 means it is false.",
        "Use only the labeled examples below. Do not assume that entity names have semantic meaning.",
        "Return JSON only, with exactly this schema:",
        '{"predictions":[{"id":0,"label":0},{"id":1,"label":1}]}',
        "",
        f"Labeled examples for run {run_idx}:",
    ]
    for _, row in train_rows.iterrows():
        label = int(row[args.label_column])
        lines.append(f"{format_pair(row, args)} -> {label}")

    lines.extend(["", "Queries to classify:"])
    for query_id, (_, row) in enumerate(test_rows.iterrows()):
        lines.append(f'id={query_id}: {format_pair(row, args)}')
    lines.append("")
    lines.append("Return one prediction for every query id. Labels must be integers 0 or 1.")
    return "\n".join(lines)


def should_use_structured_output(args: argparse.Namespace) -> bool:
    if args.structured_output == "on":
        return True
    if args.structured_output == "off":
        return False
    return args.backend == "local" and not args.model.endswith("-cloud")


def build_batches(df: pd.DataFrame, args: argparse.Namespace) -> list[PromptBatch]:
    train = df[df[args.split_column] == args.train_split].copy()
    test = df[df[args.split_column] == args.test_split].copy()
    if len(train) == 0:
        raise ValueError(f"No rows found with {args.split_column} == {args.train_split!r}")
    if len(test) == 0:
        raise ValueError(f"No rows found with {args.split_column} == {args.test_split!r}")

    batches: list[PromptBatch] = []
    for run_idx in range(args.num_runs):
        rng = random.Random(args.seed + run_idx)
        icl_rows = sample_rows(
            train,
            args.icl_size,
            rng,
            label_column=args.label_column,
            sampling=args.sampling,
        )
        test_rows = sample_rows(
            test,
            args.test_size_per_run,
            rng,
            label_column=args.label_column,
            sampling=args.sampling,
        )
        prompt = build_prompt(icl_rows, test_rows, args, run_idx)
        batches.append(
            PromptBatch(
                run_idx=run_idx,
                prompt=prompt,
                icl_indices=[int(index) for index in icl_rows.index],
                test_indices=[int(index) for index in test_rows.index],
            )
        )
    return batches


def call_ollama(prompt: str, args: argparse.Namespace) -> tuple[str, float]:
    base_url = "https://ollama.com" if args.backend == "cloud" else args.ollama_url
    url = base_url.rstrip("/") + f"/api/{args.endpoint}"
    if args.endpoint == "chat":
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": args.temperature,
                "num_predict": args.num_predict,
            },
        }
    else:
        payload = {
            "model": args.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": args.temperature,
                "num_predict": args.num_predict,
            },
        }
    if should_use_structured_output(args):
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.backend == "cloud":
        api_key = os.environ.get(args.ollama_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"--backend cloud requires ${args.ollama_api_key_env}. "
                "Create an Ollama API key and export it before running the experiment."
            )
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. "
            f"For --backend local, check `ollama serve`, `ollama signin` if using a cloud model, "
            f"and model {args.model!r}. "
            f"For --backend cloud, check ${args.ollama_api_key_env} and cloud model availability."
        ) from exc
    elapsed = time.perf_counter() - start
    if args.endpoint == "chat":
        message = response_data.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "")), elapsed
        return "", elapsed
    return str(response_data.get("response", "")), elapsed


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(0))
    array_match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
    if array_match:
        return json.loads(array_match.group(0))
    raise ValueError("No JSON object or array found in model response.")


def parse_predictions(response_text: str, expected_count: int) -> tuple[dict[int, int], str | None]:
    try:
        parsed = extract_json_object(response_text)
        if isinstance(parsed, dict):
            items = parsed.get("predictions", parsed.get("labels", parsed.get("outputs")))
        else:
            items = parsed
        if not isinstance(items, list):
            raise ValueError("JSON does not contain a predictions list.")

        predictions: dict[int, int] = {}
        for position, item in enumerate(items):
            if isinstance(item, dict):
                query_id = int(item.get("id", position))
                label = normalize_label(item.get("label", item.get("prediction")))
            else:
                query_id = position
                label = normalize_label(item)
            if 0 <= query_id < expected_count:
                predictions[query_id] = label
        return predictions, None
    except Exception as exc:  # noqa: BLE001 - stored for auditability
        return {}, str(exc)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    if not y_true:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "true_positive_rate": float("nan"),
            "predicted_positive_rate": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positive_rate": sum(y_true) / len(y_true),
        "predicted_positive_rate": sum(y_pred) / len(y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_experiment(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data_path)
    validate_columns(df, args)
    df[args.label_column] = df[args.label_column].map(normalize_label)

    batches = build_batches(df, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["data_path"] = str(args.data_path)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"data_path={args.data_path}")
        print(f"model={args.model}")
        print(f"runs={len(batches)} icl_size={args.icl_size} test_size_per_run={args.test_size_per_run}")
        print(f"first_prompt_chars={len(batches[0].prompt) if batches else 0}")
        return

    prediction_rows: list[dict[str, Any]] = []
    run_metric_rows: list[dict[str, Any]] = []
    raw_path = args.output_dir / "raw_responses.jsonl"
    raw_handle = raw_path.open("w", encoding="utf-8") if args.save_prompts else None
    try:
        for batch in batches:
            response_text, elapsed_s = call_ollama(batch.prompt, args)
            predicted_by_id, parse_error = parse_predictions(
                response_text,
                expected_count=len(batch.test_indices),
            )

            run_true: list[int] = []
            run_pred: list[int] = []
            for query_id, row_index in enumerate(batch.test_indices):
                row = df.loc[row_index]
                y_true = int(row[args.label_column])
                y_pred = predicted_by_id.get(query_id)
                valid = y_pred in (0, 1)
                if valid:
                    run_true.append(y_true)
                    run_pred.append(int(y_pred))
                prediction_rows.append(
                    {
                        "run_idx": batch.run_idx,
                        "query_id": query_id,
                        "row_index": row_index,
                        args.entity_1_column: row[args.entity_1_column],
                        args.entity_2_column: row[args.entity_2_column],
                        "true_label": y_true,
                        "predicted_label": y_pred if valid else None,
                        "is_valid_prediction": bool(valid),
                        "is_correct": bool(valid and y_true == y_pred),
                        "parse_error": parse_error,
                        "elapsed_s_for_batch": elapsed_s,
                    }
                )

            run_metrics = compute_metrics(run_true, run_pred)
            run_metrics.update(
                {
                    "run_idx": batch.run_idx,
                    "valid_predictions": len(run_true),
                    "expected_predictions": len(batch.test_indices),
                    "valid_prediction_rate": len(run_true) / len(batch.test_indices),
                    "elapsed_s": elapsed_s,
                    "parse_error": parse_error,
                }
            )
            run_metric_rows.append(run_metrics)

            if raw_handle is not None:
                raw_handle.write(
                    json.dumps(
                        {
                            "run_idx": batch.run_idx,
                            "icl_indices": batch.icl_indices,
                            "test_indices": batch.test_indices,
                            "prompt": batch.prompt,
                            "response": response_text,
                            "parse_error": parse_error,
                        }
                    )
                    + "\n"
                )
            print(
                f"run={batch.run_idx} accuracy={run_metrics['accuracy']:.4f} "
                f"valid={run_metrics['valid_predictions']}/{run_metrics['expected_predictions']} "
                f"elapsed_s={elapsed_s:.2f}"
            )
    finally:
        if raw_handle is not None:
            raw_handle.close()

    predictions_df = pd.DataFrame(prediction_rows)
    per_run_df = pd.DataFrame(run_metric_rows)
    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    per_run_df.to_csv(args.output_dir / "per_run_metrics.csv", index=False)

    valid_predictions = predictions_df[predictions_df["is_valid_prediction"]].copy()
    aggregate_metrics = compute_metrics(
        valid_predictions["true_label"].astype(int).tolist(),
        valid_predictions["predicted_label"].astype(int).tolist(),
    )
    aggregate_metrics.update(
        {
            "model": args.model,
            "data_path": str(args.data_path),
            "num_runs": args.num_runs,
            "icl_size": args.icl_size,
            "test_size_per_run": args.test_size_per_run,
            "total_expected_predictions": len(predictions_df),
            "total_valid_predictions": len(valid_predictions),
            "valid_prediction_rate": len(valid_predictions) / len(predictions_df)
            if len(predictions_df)
            else float("nan"),
            "mean_batch_elapsed_s": per_run_df["elapsed_s"].mean()
            if len(per_run_df)
            else float("nan"),
            "sampling": args.sampling,
            "reveal_symmetry": args.reveal_symmetry,
        }
    )
    pd.DataFrame([aggregate_metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Wrote results to {args.output_dir}")


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
