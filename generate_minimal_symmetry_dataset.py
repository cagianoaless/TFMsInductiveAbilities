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
            "Generate a minimal table benchmark for testing symmetry in tabular foundation models. "
            "The exported model table contains only entity_1, entity_2, label, and split."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("minimal_symmetry_dataset"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-groups", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="",
        help="Optional comma-separated group sizes. Overrides --num-groups and --group-size.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Number of sampled negative unordered pairs per positive unordered pair.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of reverse-direction holdout rows assigned to validation instead of test.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="symmetry_demonstration",
        choices=["reverse_holdout", "symmetry_demonstration"],
        help=(
            "reverse_holdout puts one direction in train and the reverse in validation/test for every pair. "
            "symmetry_demonstration additionally puts both directions of some pairs in train as ICL examples."
        ),
    )
    parser.add_argument(
        "--demo-fraction",
        type=float,
        default=0.5,
        help=(
            "Under symmetry_demonstration, fraction of unordered pairs per class used as explicit "
            "train-only symmetry demonstrations."
        ),
    )
    parser.add_argument(
        "--sequential-entity-ids",
        action="store_true",
        help=(
            "Keep entity IDs ordered by hidden group. This is useful only for debugging; by default IDs are "
            "randomized so entity number proximity does not reveal group membership."
        ),
    )
    return parser.parse_args()


def parse_group_sizes(text: str, *, num_groups: int, group_size: int) -> list[int]:
    if text.strip():
        sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        sizes = [group_size for _ in range(num_groups)]

    if len(sizes) < 2:
        raise ValueError("At least two groups are required to create negative examples.")
    if any(size < 2 for size in sizes):
        raise ValueError("Each group must contain at least two entities to create positive pairs.")
    return sizes


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_entities(
    group_sizes: list[int],
    *,
    rng: random.Random,
    sequential_entity_ids: bool,
) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    internal_records: list[dict[str, object]] = []
    internal_idx = 1
    for group_idx, group_size in enumerate(group_sizes, start=1):
        group_id = f"group_{group_idx:03d}"
        for local_idx in range(1, group_size + 1):
            internal_records.append(
                {
                    "internal_entity_id": f"internal_entity_{internal_idx:04d}",
                    "hidden_group_id": group_id,
                    "group_local_index": local_idx,
                }
            )
            internal_idx += 1

    exported_ids = [f"entity_{idx:04d}" for idx in range(1, len(internal_records) + 1)]
    if not sequential_entity_ids:
        rng.shuffle(exported_ids)

    for record, entity_id in zip(internal_records, exported_ids, strict=True):
        entities.append(
            {
                "entity_id": entity_id,
                "internal_entity_id": record["internal_entity_id"],
                "hidden_group_id": record["hidden_group_id"],
                "group_local_index": record["group_local_index"],
            }
        )
    return entities


def build_unordered_pairs(
    entities: list[dict[str, object]],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    positive_pairs: list[tuple[str, str, int]] = []
    negative_pairs: list[tuple[str, str, int]] = []

    for left_idx, left in enumerate(entities):
        for right in entities[left_idx + 1 :]:
            entity_1 = str(left["entity_id"])
            entity_2 = str(right["entity_id"])
            label = int(left["hidden_group_id"] == right["hidden_group_id"])
            if label == 1:
                positive_pairs.append((entity_1, entity_2, label))
            else:
                negative_pairs.append((entity_1, entity_2, label))

    return positive_pairs, negative_pairs


def sample_pairs(
    *,
    positive_pairs: list[tuple[str, str, int]],
    negative_pairs: list[tuple[str, str, int]],
    negative_ratio: float,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    if negative_ratio < 0.0:
        raise ValueError("--negative-ratio must be non-negative.")

    requested_negative_count = int(round(len(positive_pairs) * negative_ratio))
    negative_count = min(requested_negative_count, len(negative_pairs))
    sampled_negatives = rng.sample(negative_pairs, negative_count) if negative_count else []

    pairs = list(positive_pairs) + sampled_negatives
    if not pairs:
        raise ValueError("No unordered pairs were selected.")
    rng.shuffle(pairs)
    return pairs


def choose_demo_indices(
    unordered_pairs: list[tuple[str, str, int]],
    *,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> set[int]:
    if split_mode == "reverse_holdout":
        return set()
    if split_mode != "symmetry_demonstration":
        raise ValueError(f"Unknown split mode: {split_mode}")
    if not 0.0 <= demo_fraction < 1.0:
        raise ValueError("--demo-fraction must be in [0, 1).")

    indices_by_label: dict[int, list[int]] = {0: [], 1: []}
    for idx, (_, _, label) in enumerate(unordered_pairs):
        indices_by_label[int(label)].append(idx)

    demo_indices: set[int] = set()
    for indices in indices_by_label.values():
        if len(indices) < 2:
            continue
        demo_count = int(round(len(indices) * demo_fraction))
        demo_count = min(max(demo_count, 1), len(indices) - 1)
        demo_indices.update(rng.sample(indices, demo_count))
    return demo_indices


def assign_symmetry_split(
    *,
    unordered_pairs: list[tuple[str, str, int]],
    validation_fraction: float,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")

    demo_indices = choose_demo_indices(
        unordered_pairs,
        split_mode=split_mode,
        demo_fraction=demo_fraction,
        rng=rng,
    )
    query_indices = [idx for idx in range(len(unordered_pairs)) if idx not in demo_indices]
    if not query_indices:
        raise ValueError("No query pairs remain after selecting demonstration pairs.")
    reverse_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))
    atom_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for pair_idx, (left, right, label) in enumerate(unordered_pairs, start=1):
        zero_based_pair_idx = pair_idx - 1
        if rng.random() < 0.5:
            train_left, train_right = left, right
            holdout_left, holdout_right = right, left
        else:
            train_left, train_right = right, left
            holdout_left, holdout_right = left, right

        pair_id = f"pair_{pair_idx:06d}"
        pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
        holdout_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in reverse_indices else "test"
        atom_rows.append(
            {
                "entity_1": train_left,
                "entity_2": train_right,
                "label": label,
                "split": "train",
            }
        )
        atom_rows.append(
            {
                "entity_1": holdout_left,
                "entity_2": holdout_right,
                "label": label,
                "split": holdout_split,
            }
        )

        pair_rows.append(
            {
                "pair_id": pair_id,
                "pair_role": pair_role,
                "unordered_entity_a": left,
                "unordered_entity_b": right,
                "label": label,
                "train_entity_1": train_left,
                "train_entity_2": train_right,
                "holdout_entity_1": holdout_left,
                "holdout_entity_2": holdout_right,
                "reverse_split": holdout_split,
            }
        )

    return atom_rows, pair_rows


def validate_dataset(atom_rows: list[dict[str, object]]) -> dict[str, object]:
    required_columns = {"entity_1", "entity_2", "label", "split"}
    actual_columns = set(atom_rows[0]) if atom_rows else set()
    forbidden_feature_columns = {"group_id", "hidden_group_id", "entity_1_group", "entity_2_group"}
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
    reverse_label_mismatch_examples: list[tuple[str, str]] = []
    for (entity_1, entity_2), label in labels_by_direction.items():
        reverse_label = labels_by_direction.get((entity_2, entity_1))
        if reverse_label is None:
            missing_reverse_examples.append((entity_1, entity_2))
        elif reverse_label != label:
            reverse_label_mismatch_examples.append((entity_1, entity_2))

    return {
        "has_required_columns": required_columns <= actual_columns,
        "forbidden_feature_columns_present": sorted(forbidden_feature_columns & actual_columns),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": dict(sorted(split_label_counts.items())),
        "missing_reverse_count": len(missing_reverse_examples),
        "reverse_label_mismatch_count": len(reverse_label_mismatch_examples),
        "missing_reverse_examples": missing_reverse_examples[:10],
        "reverse_label_mismatch_examples": reverse_label_mismatch_examples[:10],
        "is_valid_symmetry_holdout": (
            required_columns <= actual_columns
            and not (forbidden_feature_columns & actual_columns)
            and len(missing_reverse_examples) == 0
            and len(reverse_label_mismatch_examples) == 0
            and split_counts.get("train", 0) > 0
            and split_counts.get("test", 0) > 0
        ),
    }


def build_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation"]
    lines = [
        "# Minimal Symmetry Dataset",
        "",
        "The model-facing table is `symmetry_friendship_atoms.csv`.",
        "",
        "Use only `entity_1` and `entity_2` as features and `label` as the target.",
        "`split` is metadata for train/validation/test selection, not a model feature.",
        "",
        "## Shape",
        "",
        f"- entities: `{metadata['num_entities']}`",
        f"- hidden groups: `{metadata['num_groups']}`",
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
    lines.append(f"- symmetry dataset valid: `{validation['is_valid_symmetry_holdout']}`")
    lines.append(f"- hidden group columns in model table: `{validation['forbidden_feature_columns_present']}`")
    lines.append(f"- reverse label mismatches: `{validation['reverse_label_mismatch_count']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    group_sizes = parse_group_sizes(args.group_sizes, num_groups=args.num_groups, group_size=args.group_size)
    rng = random.Random(args.seed)

    entities = build_entities(
        group_sizes,
        rng=rng,
        sequential_entity_ids=args.sequential_entity_ids,
    )
    positive_pairs, negative_pairs = build_unordered_pairs(entities)
    unordered_pairs = sample_pairs(
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )
    atom_rows, pair_rows = assign_symmetry_split(
        unordered_pairs=unordered_pairs,
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
        "num_groups": len(group_sizes),
        "group_sizes": group_sizes,
        "num_entities": len(entities),
        "num_positive_unordered_pairs_available": len(positive_pairs),
        "num_negative_unordered_pairs_available": len(negative_pairs),
        "num_unordered_pairs": len(unordered_pairs),
        "num_atom_rows": len(atom_rows),
        "negative_ratio": args.negative_ratio,
        "validation_fraction": args.validation_fraction,
        "split_mode": args.split_mode,
        "demo_fraction": args.demo_fraction,
        "pair_role_counts": dict(sorted(pair_role_counts.items())),
        "entity_ids_randomized": not args.sequential_entity_ids,
        "generation_rule": (
            "label is 1 when the two entities are in the same hidden group, else 0. "
            "Hidden groups are never written to the model-facing atom table."
        ),
        "split_rule": (
            "reverse_holdout: one direction is train and the reverse direction is validation/test. "
            "symmetry_demonstration: demo pairs have both directions in train; query pairs use one direction "
            "in train and the reverse direction in validation/test."
        ),
        "validation": validation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "symmetry_friendship_atoms.csv",
        atom_rows,
        ["entity_1", "entity_2", "label", "split"],
    )
    write_csv(
        output_dir / "entities.csv",
        entities,
        ["entity_id", "internal_entity_id", "hidden_group_id", "group_local_index"],
    )
    write_csv(
        output_dir / "pair_assignments.csv",
        pair_rows,
        [
            "pair_id",
            "pair_role",
            "unordered_entity_a",
            "unordered_entity_b",
            "label",
            "train_entity_1",
            "train_entity_2",
            "holdout_entity_1",
            "holdout_entity_2",
            "reverse_split",
        ],
    )
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "validation_report.md").write_text(build_report(metadata), encoding="utf-8")

    if not validation["is_valid_symmetry_holdout"]:
        raise RuntimeError(f"Generated dataset failed validation. See {output_dir / 'metadata.json'}")

    print(f"Wrote minimal symmetry dataset to {output_dir}")
    print(
        "entities={entities} rows={rows} train={train} validation={validation} test={test}".format(
            entities=len(entities),
            rows=len(atom_rows),
            train=validation["split_counts"].get("train", 0),
            validation=validation["split_counts"].get("validation", 0),
            test=validation["split_counts"].get("test", 0),
        )
    )
    print(f"label_counts={validation['label_counts']}")


if __name__ == "__main__":
    main()
