#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a minimal table benchmark for antisymmetry. The academic example is "
            "'senior_coauthor_of': if A is senior to B, then B is not senior to A."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("minimal_antisymmetry_dataset"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-entities", type=int, default=60, help="Number of synthetic authors.")
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=240,
        help="Number of unordered author pairs to sample before creating both directions.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of query reverse-direction rows assigned to validation instead of test.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="antisymmetry_demonstration",
        choices=["reverse_holdout", "antisymmetry_demonstration"],
        help=(
            "reverse_holdout puts one direction in train and the opposite direction in validation/test. "
            "antisymmetry_demonstration additionally puts both directions of some pairs in train as ICL examples."
        ),
    )
    parser.add_argument(
        "--demo-fraction",
        type=float,
        default=0.5,
        help=(
            "Under antisymmetry_demonstration, fraction of unordered pairs used as explicit train-only "
            "antisymmetry demonstrations."
        ),
    )
    parser.add_argument(
        "--sequential-entity-ids",
        action="store_true",
        help=(
            "Keep exported author IDs ordered by hidden seniority. This is for debugging only; by default IDs "
            "are randomized so entity number does not reveal seniority."
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_authors(
    *,
    num_entities: int,
    rng: random.Random,
    sequential_entity_ids: bool,
) -> list[dict[str, object]]:
    if num_entities < 2:
        raise ValueError("--num-entities must be at least 2.")

    internal_records = [
        {
            "internal_entity_id": f"internal_author_{idx:04d}",
            "hidden_seniority_rank": idx,
        }
        for idx in range(1, num_entities + 1)
    ]
    exported_ids = [f"entity_{idx:04d}" for idx in range(1, num_entities + 1)]
    if not sequential_entity_ids:
        rng.shuffle(exported_ids)

    authors: list[dict[str, object]] = []
    for record, entity_id in zip(internal_records, exported_ids, strict=True):
        authors.append(
            {
                "entity_id": entity_id,
                "internal_entity_id": record["internal_entity_id"],
                "hidden_seniority_rank": record["hidden_seniority_rank"],
            }
        )
    return authors


def build_unordered_pairs(authors: list[dict[str, object]]) -> list[tuple[str, str, int, int]]:
    pairs: list[tuple[str, str, int, int]] = []
    for left_idx, left in enumerate(authors):
        for right in authors[left_idx + 1 :]:
            pairs.append(
                (
                    str(left["entity_id"]),
                    str(right["entity_id"]),
                    int(left["hidden_seniority_rank"]),
                    int(right["hidden_seniority_rank"]),
                )
            )
    return pairs


def sample_pairs(
    *,
    pairs: list[tuple[str, str, int, int]],
    num_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str, int, int]]:
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive.")
    sample_count = min(num_pairs, len(pairs))
    sampled = rng.sample(pairs, sample_count)
    rng.shuffle(sampled)
    return sampled


def choose_demo_indices(
    sampled_pairs: list[tuple[str, str, int, int]],
    *,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> set[int]:
    if split_mode == "reverse_holdout":
        return set()
    if split_mode != "antisymmetry_demonstration":
        raise ValueError(f"Unknown split mode: {split_mode}")
    if not 0.0 <= demo_fraction < 1.0:
        raise ValueError("--demo-fraction must be in [0, 1).")

    if len(sampled_pairs) < 2:
        return set()
    demo_count = int(round(len(sampled_pairs) * demo_fraction))
    demo_count = min(max(demo_count, 1), len(sampled_pairs) - 1)
    return set(rng.sample(range(len(sampled_pairs)), demo_count))


def directed_labels(
    left: str,
    right: str,
    left_rank: int,
    right_rank: int,
) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    if left_rank < right_rank:
        return (left, right, 1), (right, left, 0)
    return (right, left, 1), (left, right, 0)


def assign_antisymmetry_split(
    *,
    sampled_pairs: list[tuple[str, str, int, int]],
    validation_fraction: float,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")

    demo_indices = choose_demo_indices(
        sampled_pairs,
        split_mode=split_mode,
        demo_fraction=demo_fraction,
        rng=rng,
    )
    query_indices = [idx for idx in range(len(sampled_pairs)) if idx not in demo_indices]
    if not query_indices:
        raise ValueError("No query pairs remain after selecting demonstration pairs.")
    validation_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))

    atom_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for pair_idx, (left, right, left_rank, right_rank) in enumerate(sampled_pairs, start=1):
        positive_direction, negative_direction = directed_labels(left, right, left_rank, right_rank)
        if rng.random() < 0.5:
            train_direction = positive_direction
            reverse_direction = negative_direction
        else:
            train_direction = negative_direction
            reverse_direction = positive_direction

        zero_based_pair_idx = pair_idx - 1
        pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
        reverse_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in validation_indices else "test"
        pair_id = f"pair_{pair_idx:06d}"

        atom_rows.append(
            {
                "entity_1": train_direction[0],
                "entity_2": train_direction[1],
                "label": train_direction[2],
                "split": "train",
            }
        )
        atom_rows.append(
            {
                "entity_1": reverse_direction[0],
                "entity_2": reverse_direction[1],
                "label": reverse_direction[2],
                "split": reverse_split,
            }
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "pair_role": pair_role,
                "unordered_entity_a": left,
                "unordered_entity_b": right,
                "positive_entity_1": positive_direction[0],
                "positive_entity_2": positive_direction[1],
                "negative_entity_1": negative_direction[0],
                "negative_entity_2": negative_direction[1],
                "train_entity_1": train_direction[0],
                "train_entity_2": train_direction[1],
                "train_label": train_direction[2],
                "reverse_entity_1": reverse_direction[0],
                "reverse_entity_2": reverse_direction[1],
                "reverse_label": reverse_direction[2],
                "reverse_split": reverse_split,
            }
        )

    return atom_rows, pair_rows


def validate_dataset(atom_rows: list[dict[str, object]]) -> dict[str, object]:
    required_columns = {"entity_1", "entity_2", "label", "split"}
    actual_columns = set(atom_rows[0]) if atom_rows else set()
    forbidden_feature_columns = {"hidden_seniority_rank", "rank", "seniority", "internal_entity_id"}
    labels_by_direction: dict[tuple[str, str], int] = {}
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: Counter[str] = Counter()

    for row in atom_rows:
        split = str(row["split"])
        label = int(row["label"])
        entity_1 = str(row["entity_1"])
        entity_2 = str(row["entity_2"])
        labels_by_direction[(entity_1, entity_2)] = label
        split_counts[split] += 1
        label_counts[str(label)] += 1
        split_label_counts[f"{split}:{label}"] += 1

    missing_reverse_examples: list[tuple[str, str]] = []
    antisymmetry_violation_examples: list[tuple[str, str]] = []
    for (entity_1, entity_2), label in labels_by_direction.items():
        reverse_label = labels_by_direction.get((entity_2, entity_1))
        if reverse_label is None:
            missing_reverse_examples.append((entity_1, entity_2))
        elif reverse_label == label:
            antisymmetry_violation_examples.append((entity_1, entity_2))

    return {
        "has_required_columns": required_columns <= actual_columns,
        "forbidden_feature_columns_present": sorted(forbidden_feature_columns & actual_columns),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": dict(sorted(split_label_counts.items())),
        "missing_reverse_count": len(missing_reverse_examples),
        "antisymmetry_violation_count": len(antisymmetry_violation_examples),
        "missing_reverse_examples": missing_reverse_examples[:10],
        "antisymmetry_violation_examples": antisymmetry_violation_examples[:10],
        "is_valid_antisymmetry_holdout": (
            required_columns <= actual_columns
            and not (forbidden_feature_columns & actual_columns)
            and len(missing_reverse_examples) == 0
            and len(antisymmetry_violation_examples) == 0
            and split_counts.get("train", 0) > 0
            and split_counts.get("test", 0) > 0
        ),
    }


def build_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation"]
    lines = [
        "# Minimal Antisymmetry Dataset",
        "",
        "The model-facing table is `antisymmetry_coauthor_atoms.csv`.",
        "",
        "Relation: `senior_coauthor_of`. This is directed and antisymmetric: if `A -> B = 1`, then `B -> A = 0`.",
        "",
        "Use only `entity_1` and `entity_2` as features and `label` as the target.",
        "`split` is metadata for train/validation/test selection, not a model feature.",
        "",
        "## Shape",
        "",
        f"- entities: `{metadata['num_entities']}`",
        f"- sampled unordered pairs: `{metadata['num_unordered_pairs']}`",
        f"- atom rows: `{metadata['num_atom_rows']}`",
        f"- split mode: `{metadata['split_mode']}`",
        f"- demo unordered pairs: `{metadata['pair_role_counts'].get('demo', 0)}`",
        f"- query unordered pairs: `{metadata['pair_role_counts'].get('query', 0)}`",
        "",
        "## Split Counts",
        "",
    ]
    for split, count in validation["split_counts"].items():
        lines.append(f"- `{split}`: `{count}`")

    lines.extend(["", "## Validation", ""])
    lines.append(f"- antisymmetry dataset valid: `{validation['is_valid_antisymmetry_holdout']}`")
    lines.append(f"- hidden rank columns in model table: `{validation['forbidden_feature_columns_present']}`")
    lines.append(f"- antisymmetry violations: `{validation['antisymmetry_violation_count']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    authors = build_authors(
        num_entities=args.num_entities,
        rng=rng,
        sequential_entity_ids=args.sequential_entity_ids,
    )
    all_pairs = build_unordered_pairs(authors)
    sampled_pairs = sample_pairs(pairs=all_pairs, num_pairs=args.num_pairs, rng=rng)
    atom_rows, pair_rows = assign_antisymmetry_split(
        sampled_pairs=sampled_pairs,
        validation_fraction=args.validation_fraction,
        split_mode=args.split_mode,
        demo_fraction=args.demo_fraction,
        rng=rng,
    )
    validation = validate_dataset(atom_rows)
    pair_role_counts = Counter(str(row["pair_role"]) for row in pair_rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "relation_name": "senior_coauthor_of",
        "num_entities": len(authors),
        "num_unordered_pairs_available": len(all_pairs),
        "num_unordered_pairs": len(sampled_pairs),
        "num_atom_rows": len(atom_rows),
        "validation_fraction": args.validation_fraction,
        "split_mode": args.split_mode,
        "demo_fraction": args.demo_fraction,
        "pair_role_counts": dict(sorted(pair_role_counts.items())),
        "entity_ids_randomized": not args.sequential_entity_ids,
        "generation_rule": (
            "Each author has a hidden seniority rank. label=1 means entity_1 is more senior than entity_2. "
            "Hidden seniority is never written to the model-facing atom table."
        ),
        "split_rule": (
            "reverse_holdout: one direction is train and the opposite-label reverse direction is validation/test. "
            "antisymmetry_demonstration: demo pairs have both directions in train; query pairs use one direction "
            "in train and the reverse direction in validation/test."
        ),
        "validation": validation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "antisymmetry_coauthor_atoms.csv",
        atom_rows,
        ["entity_1", "entity_2", "label", "split"],
    )
    write_csv(
        output_dir / "entities.csv",
        authors,
        ["entity_id", "internal_entity_id", "hidden_seniority_rank"],
    )
    write_csv(
        output_dir / "pair_assignments.csv",
        pair_rows,
        [
            "pair_id",
            "pair_role",
            "unordered_entity_a",
            "unordered_entity_b",
            "positive_entity_1",
            "positive_entity_2",
            "negative_entity_1",
            "negative_entity_2",
            "train_entity_1",
            "train_entity_2",
            "train_label",
            "reverse_entity_1",
            "reverse_entity_2",
            "reverse_label",
            "reverse_split",
        ],
    )
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "validation_report.md").write_text(build_report(metadata), encoding="utf-8")

    if not validation["is_valid_antisymmetry_holdout"]:
        raise RuntimeError(f"Generated dataset failed validation. See {output_dir / 'metadata.json'}")

    print(f"Wrote minimal antisymmetry dataset to {output_dir}")
    print(
        "entities={entities} rows={rows} train={train} validation={validation} test={test}".format(
            entities=len(authors),
            rows=len(atom_rows),
            train=validation["split_counts"].get("train", 0),
            validation=validation["split_counts"].get("validation", 0),
            test=validation["split_counts"].get("test", 0),
        )
    )
    print(f"label_counts={validation['label_counts']}")


if __name__ == "__main__":
    main()
