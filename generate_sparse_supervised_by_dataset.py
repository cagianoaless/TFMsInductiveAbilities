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
            "Generate a sparse supervised_by relation table. Positive rows mean entity_1 is supervised by "
            "entity_2. The model-facing CSV contains only entity_1, entity_2, label, and split."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sparse_supervised_by_dataset"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-students", type=int, default=48)
    parser.add_argument("--num-supervisors", type=int, default=12)
    parser.add_argument("--min-supervisors-per-student", type=int, default=1)
    parser.add_argument("--max-supervisors-per-student", type=int, default=2)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help=(
            "Number of sampled both-negative unordered pairs per positive supervision pair. "
            "Higher values make the dataset sparser and more realistic."
        ),
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
        default="supervision_demonstration",
        choices=["reverse_holdout", "supervision_demonstration"],
        help=(
            "reverse_holdout puts one direction in train and the reverse in validation/test for every pair. "
            "supervision_demonstration additionally puts both directions of some pairs in train as ICL examples."
        ),
    )
    parser.add_argument(
        "--demo-fraction",
        type=float,
        default=0.5,
        help=(
            "Under supervision_demonstration, fraction of unordered pairs per pair type used as explicit "
            "train-only reverse-direction demonstrations."
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


def build_entities(
    *,
    num_students: int,
    num_supervisors: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    if num_students < 2:
        raise ValueError("--num-students must be at least 2.")
    if num_supervisors < 1:
        raise ValueError("--num-supervisors must be at least 1.")

    internal_records: list[dict[str, object]] = []
    for idx in range(1, num_students + 1):
        internal_records.append(
            {
                "internal_entity_id": f"internal_student_{idx:04d}",
                "role": "student",
            }
        )
    for idx in range(1, num_supervisors + 1):
        internal_records.append(
            {
                "internal_entity_id": f"internal_supervisor_{idx:04d}",
                "role": "supervisor",
            }
        )

    exported_ids = [f"entity_{idx:04d}" for idx in range(1, len(internal_records) + 1)]
    rng.shuffle(exported_ids)

    return [
        {
            "entity_id": entity_id,
            "internal_entity_id": record["internal_entity_id"],
            "role": record["role"],
        }
        for record, entity_id in zip(internal_records, exported_ids, strict=True)
    ]


def all_unordered_pairs(entity_ids: list[str]) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left_idx, left in enumerate(entity_ids)
        for right in entity_ids[left_idx + 1 :]
    ]


def sample_supervision_edges(
    *,
    entities: list[dict[str, object]],
    min_supervisors_per_student: int,
    max_supervisors_per_student: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    if min_supervisors_per_student < 1:
        raise ValueError("--min-supervisors-per-student must be at least 1.")
    if max_supervisors_per_student < min_supervisors_per_student:
        raise ValueError("--max-supervisors-per-student must be >= --min-supervisors-per-student.")

    students = [str(row["entity_id"]) for row in entities if row["role"] == "student"]
    supervisors = [str(row["entity_id"]) for row in entities if row["role"] == "supervisor"]
    if max_supervisors_per_student > len(supervisors):
        raise ValueError("--max-supervisors-per-student cannot exceed --num-supervisors.")

    positive_edges: list[tuple[str, str]] = []
    for student in students:
        supervisor_count = rng.randint(min_supervisors_per_student, max_supervisors_per_student)
        for supervisor in rng.sample(supervisors, supervisor_count):
            positive_edges.append((student, supervisor))
    rng.shuffle(positive_edges)
    return positive_edges


def build_pair_records(
    *,
    entities: list[dict[str, object]],
    positive_edges: list[tuple[str, str]],
    negative_ratio: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    if negative_ratio < 0.0:
        raise ValueError("--negative-ratio must be non-negative.")

    entity_ids = [str(row["entity_id"]) for row in entities]
    positive_lookup = set(positive_edges)
    positive_unordered_lookup = {frozenset(edge) for edge in positive_edges}

    positive_pair_records: list[dict[str, object]] = []
    negative_pair_records: list[dict[str, object]] = []

    for left, right in all_unordered_pairs(entity_ids):
        unordered_key = frozenset((left, right))
        if unordered_key in positive_unordered_lookup:
            if (left, right) in positive_lookup:
                label_left_to_right = 1
                label_right_to_left = 0
                student_entity = left
                supervisor_entity = right
            elif (right, left) in positive_lookup:
                label_left_to_right = 0
                label_right_to_left = 1
                student_entity = right
                supervisor_entity = left
            else:
                raise RuntimeError("Positive unordered lookup is inconsistent with directed edges.")

            positive_pair_records.append(
                {
                    "entity_a": left,
                    "entity_b": right,
                    "label_a_to_b": label_left_to_right,
                    "label_b_to_a": label_right_to_left,
                    "pair_type": "supervision_edge",
                    "student_entity": student_entity,
                    "supervisor_entity": supervisor_entity,
                }
            )
        else:
            negative_pair_records.append(
                {
                    "entity_a": left,
                    "entity_b": right,
                    "label_a_to_b": 0,
                    "label_b_to_a": 0,
                    "pair_type": "unrelated_pair",
                    "student_entity": "",
                    "supervisor_entity": "",
                }
            )

    requested_negative_count = int(round(len(positive_pair_records) * negative_ratio))
    negative_count = min(requested_negative_count, len(negative_pair_records))
    sampled_negatives = rng.sample(negative_pair_records, negative_count) if negative_count else []

    pair_records = positive_pair_records + sampled_negatives
    if not positive_pair_records:
        raise ValueError("No positive supervision edges were generated.")
    if not sampled_negatives:
        raise ValueError("No negative unrelated pairs were sampled.")
    rng.shuffle(pair_records)
    return pair_records


def choose_demo_indices(
    pair_records: list[dict[str, object]],
    *,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> set[int]:
    if split_mode == "reverse_holdout":
        return set()
    if split_mode != "supervision_demonstration":
        raise ValueError(f"Unknown split mode: {split_mode}")
    if not 0.0 <= demo_fraction < 1.0:
        raise ValueError("--demo-fraction must be in [0, 1).")

    indices_by_type: dict[str, list[int]] = {
        "supervision_edge": [],
        "unrelated_pair": [],
    }
    for idx, row in enumerate(pair_records):
        indices_by_type[str(row["pair_type"])].append(idx)

    demo_indices: set[int] = set()
    for indices in indices_by_type.values():
        if len(indices) < 2:
            continue
        demo_count = int(round(len(indices) * demo_fraction))
        demo_count = min(max(demo_count, 1), len(indices) - 1)
        demo_indices.update(rng.sample(indices, demo_count))
    return demo_indices


def assign_supervision_split(
    *,
    pair_records: list[dict[str, object]],
    validation_fraction: float,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")

    demo_indices = choose_demo_indices(
        pair_records,
        split_mode=split_mode,
        demo_fraction=demo_fraction,
        rng=rng,
    )
    query_indices = [idx for idx in range(len(pair_records)) if idx not in demo_indices]
    if not query_indices:
        raise ValueError("No query pairs remain after selecting demonstration pairs.")
    validation_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))

    atom_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    for pair_idx, pair_record in enumerate(pair_records, start=1):
        zero_based_pair_idx = pair_idx - 1
        entity_a = str(pair_record["entity_a"])
        entity_b = str(pair_record["entity_b"])
        label_a_to_b = int(pair_record["label_a_to_b"])
        label_b_to_a = int(pair_record["label_b_to_a"])

        if rng.random() < 0.5:
            train_entity_1 = entity_a
            train_entity_2 = entity_b
            train_label = label_a_to_b
            reverse_entity_1 = entity_b
            reverse_entity_2 = entity_a
            reverse_label = label_b_to_a
        else:
            train_entity_1 = entity_b
            train_entity_2 = entity_a
            train_label = label_b_to_a
            reverse_entity_1 = entity_a
            reverse_entity_2 = entity_b
            reverse_label = label_a_to_b

        pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
        reverse_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in validation_indices else "test"
        pair_id = f"pair_{pair_idx:06d}"

        atom_rows.append(
            {
                "entity_1": train_entity_1,
                "entity_2": train_entity_2,
                "label": train_label,
                "split": "train",
            }
        )
        atom_rows.append(
            {
                "entity_1": reverse_entity_1,
                "entity_2": reverse_entity_2,
                "label": reverse_label,
                "split": reverse_split,
            }
        )
        assignment_rows.append(
            {
                "pair_id": pair_id,
                "pair_role": pair_role,
                "pair_type": pair_record["pair_type"],
                "unordered_entity_a": entity_a,
                "unordered_entity_b": entity_b,
                "label_a_to_b": label_a_to_b,
                "label_b_to_a": label_b_to_a,
                "student_entity": pair_record["student_entity"],
                "supervisor_entity": pair_record["supervisor_entity"],
                "train_entity_1": train_entity_1,
                "train_entity_2": train_entity_2,
                "train_label": train_label,
                "reverse_entity_1": reverse_entity_1,
                "reverse_entity_2": reverse_entity_2,
                "reverse_label": reverse_label,
                "reverse_split": reverse_split,
            }
        )

    return atom_rows, assignment_rows


def validate_dataset(atom_rows: list[dict[str, object]]) -> dict[str, object]:
    required_columns = {"entity_1", "entity_2", "label", "split"}
    actual_columns = set(atom_rows[0]) if atom_rows else set()
    forbidden_feature_columns = {
        "role",
        "student",
        "supervisor",
        "student_entity",
        "supervisor_entity",
        "internal_entity_id",
    }
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
    both_positive_examples: list[tuple[str, str]] = []
    positive_reverse_negative_count = 0
    both_negative_pair_count = 0
    visited_unordered_pairs: set[frozenset[str]] = set()

    for (entity_1, entity_2), label in labels_by_direction.items():
        reverse_label = labels_by_direction.get((entity_2, entity_1))
        if reverse_label is None:
            missing_reverse_examples.append((entity_1, entity_2))
            continue
        if label == 1 and reverse_label == 1:
            both_positive_examples.append((entity_1, entity_2))
        if label == 1 and reverse_label == 0:
            positive_reverse_negative_count += 1

        unordered_key = frozenset((entity_1, entity_2))
        if unordered_key not in visited_unordered_pairs:
            visited_unordered_pairs.add(unordered_key)
            if label == 0 and reverse_label == 0:
                both_negative_pair_count += 1

    train_labels = {
        int(row["label"])
        for row in atom_rows
        if row["split"] == "train"
    }
    test_labels = {
        int(row["label"])
        for row in atom_rows
        if row["split"] == "test"
    }

    return {
        "has_required_columns": required_columns <= actual_columns,
        "forbidden_feature_columns_present": sorted(forbidden_feature_columns & actual_columns),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": dict(sorted(split_label_counts.items())),
        "missing_reverse_count": len(missing_reverse_examples),
        "both_positive_violation_count": len(both_positive_examples),
        "positive_reverse_negative_count": positive_reverse_negative_count,
        "both_negative_unordered_pair_count": both_negative_pair_count,
        "missing_reverse_examples": missing_reverse_examples[:10],
        "both_positive_violation_examples": both_positive_examples[:10],
        "positive_rate": (
            label_counts.get("1", 0) / sum(label_counts.values())
            if label_counts
            else 0.0
        ),
        "is_valid_sparse_supervised_by": (
            required_columns <= actual_columns
            and not (forbidden_feature_columns & actual_columns)
            and len(missing_reverse_examples) == 0
            and len(both_positive_examples) == 0
            and positive_reverse_negative_count > 0
            and both_negative_pair_count > 0
            and split_counts.get("train", 0) > 0
            and split_counts.get("test", 0) > 0
            and train_labels == {0, 1}
            and test_labels == {0, 1}
        ),
    }


def build_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation"]
    lines = [
        "# Sparse Supervised-By Dataset",
        "",
        "The model-facing table is `supervised_by_atoms.csv`.",
        "",
        "`label=1` means `entity_1` is supervised by `entity_2`.",
        "The relation is sparse: only sampled student-supervisor edges are positive.",
        "Unrelated pairs are negative in both directions, which is different from the old total-order antisymmetry dataset.",
        "",
        "Use only `entity_1` and `entity_2` as features and `label` as the target.",
        "`split` is metadata for train/validation/test selection, not a model feature.",
        "",
        "## Shape",
        "",
        f"- students: `{metadata['num_students']}`",
        f"- supervisors: `{metadata['num_supervisors']}`",
        f"- positive supervision edges: `{metadata['num_positive_supervision_edges']}`",
        f"- sampled unrelated unordered pairs: `{metadata['num_sampled_unrelated_pairs']}`",
        f"- atom rows: `{metadata['num_atom_rows']}`",
        f"- positive rate: `{validation['positive_rate']:.4f}`",
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
    lines.append(f"- sparse supervised_by dataset valid: `{validation['is_valid_sparse_supervised_by']}`")
    lines.append(f"- both-positive antisymmetry violations: `{validation['both_positive_violation_count']}`")
    lines.append(f"- positive rows whose reverse is negative: `{validation['positive_reverse_negative_count']}`")
    lines.append(f"- both-negative unordered pairs: `{validation['both_negative_unordered_pair_count']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    entities = build_entities(
        num_students=args.num_students,
        num_supervisors=args.num_supervisors,
        rng=rng,
    )
    positive_edges = sample_supervision_edges(
        entities=entities,
        min_supervisors_per_student=args.min_supervisors_per_student,
        max_supervisors_per_student=args.max_supervisors_per_student,
        rng=rng,
    )
    pair_records = build_pair_records(
        entities=entities,
        positive_edges=positive_edges,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )
    atom_rows, pair_rows = assign_supervision_split(
        pair_records=pair_records,
        validation_fraction=args.validation_fraction,
        split_mode=args.split_mode,
        demo_fraction=args.demo_fraction,
        rng=rng,
    )
    validation = validate_dataset(atom_rows)
    pair_role_counts = Counter(str(row["pair_role"]) for row in pair_rows)
    pair_type_counts = Counter(str(row["pair_type"]) for row in pair_rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "relation_name": "supervised_by",
        "num_students": args.num_students,
        "num_supervisors": args.num_supervisors,
        "num_entities": len(entities),
        "num_positive_supervision_edges": len(positive_edges),
        "num_sampled_unrelated_pairs": pair_type_counts.get("unrelated_pair", 0),
        "num_unordered_pairs": len(pair_rows),
        "num_atom_rows": len(atom_rows),
        "min_supervisors_per_student": args.min_supervisors_per_student,
        "max_supervisors_per_student": args.max_supervisors_per_student,
        "negative_ratio": args.negative_ratio,
        "validation_fraction": args.validation_fraction,
        "split_mode": args.split_mode,
        "demo_fraction": args.demo_fraction,
        "pair_role_counts": dict(sorted(pair_role_counts.items())),
        "pair_type_counts": dict(sorted(pair_type_counts.items())),
        "entity_ids_randomized": True,
        "generation_rule": (
            "Students are assigned one or more hidden supervisors. label=1 means entity_1 is supervised by "
            "entity_2. Reverse supervision rows are 0. Unrelated pairs are 0 in both directions. Roles are "
            "written only to audit files and never to the model-facing atom table."
        ),
        "split_rule": (
            "reverse_holdout: one direction is train and the reverse direction is validation/test. "
            "supervision_demonstration: demo pairs have both directions in train; query pairs use one direction "
            "in train and the reverse direction in validation/test."
        ),
        "validation": validation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "supervised_by_atoms.csv",
        atom_rows,
        ["entity_1", "entity_2", "label", "split"],
    )
    write_csv(
        output_dir / "entities.csv",
        entities,
        ["entity_id", "internal_entity_id", "role"],
    )
    write_csv(
        output_dir / "positive_supervisions.csv",
        [
            {
                "edge_id": f"edge_{idx:06d}",
                "student_entity": student,
                "supervisor_entity": supervisor,
            }
            for idx, (student, supervisor) in enumerate(positive_edges, start=1)
        ],
        ["edge_id", "student_entity", "supervisor_entity"],
    )
    write_csv(
        output_dir / "pair_assignments.csv",
        pair_rows,
        [
            "pair_id",
            "pair_role",
            "pair_type",
            "unordered_entity_a",
            "unordered_entity_b",
            "label_a_to_b",
            "label_b_to_a",
            "student_entity",
            "supervisor_entity",
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

    if not validation["is_valid_sparse_supervised_by"]:
        raise RuntimeError(f"Generated dataset failed validation. See {output_dir / 'metadata.json'}")

    print(f"Wrote sparse supervised_by dataset to {output_dir}")
    print(
        "entities={entities} supervision_edges={edges} unrelated_pairs={unrelated} rows={rows} "
        "train={train} validation={validation} test={test}".format(
            entities=len(entities),
            edges=len(positive_edges),
            unrelated=pair_type_counts.get("unrelated_pair", 0),
            rows=len(atom_rows),
            train=validation["split_counts"].get("train", 0),
            validation=validation["split_counts"].get("validation", 0),
            test=validation["split_counts"].get("test", 0),
        )
    )
    print(f"label_counts={validation['label_counts']}")
    print(f"positive_rate={validation['positive_rate']:.4f}")


if __name__ == "__main__":
    main()
