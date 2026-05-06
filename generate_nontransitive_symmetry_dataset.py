#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a symmetric but non-transitive binary relation table. "
            "The model-facing CSV contains only entity_1, entity_2, label, and split."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("nontransitive_symmetry_dataset"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-entities", type=int, default=80)
    parser.add_argument(
        "--positive-degree",
        type=int,
        default=4,
        help="Approximate number of positive undirected edges incident to each entity.",
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
        help="Fraction of query reverse-direction rows assigned to validation instead of test.",
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
        "--max-generation-attempts",
        type=int,
        default=200,
        help="Maximum attempts to sample a connected-enough non-transitive graph before failing.",
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


def build_entities(num_entities: int, *, rng: random.Random) -> list[dict[str, object]]:
    if num_entities < 4:
        raise ValueError("--num-entities must be at least 4.")

    exported_ids = [f"entity_{idx:04d}" for idx in range(1, num_entities + 1)]
    rng.shuffle(exported_ids)
    return [
        {
            "entity_id": entity_id,
            "internal_entity_id": f"internal_entity_{idx:04d}",
        }
        for idx, entity_id in enumerate(exported_ids, start=1)
    ]


def all_unordered_pairs(entity_ids: list[str]) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left_idx, left in enumerate(entity_ids)
        for right in entity_ids[left_idx + 1 :]
    ]


def sample_positive_edges(
    *,
    unordered_pairs: list[tuple[str, str]],
    num_entities: int,
    positive_degree: int,
    rng: random.Random,
) -> set[tuple[str, str]]:
    if positive_degree <= 0:
        raise ValueError("--positive-degree must be positive.")

    max_edges = len(unordered_pairs)
    requested_edges = int(round((num_entities * positive_degree) / 2))
    edge_count = min(max(requested_edges, 2), max_edges - 1)
    return set(rng.sample(unordered_pairs, edge_count))


def adjacency_from_edges(positive_edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in positive_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def count_transitivity_violations(
    *,
    entity_ids: list[str],
    positive_edges: set[tuple[str, str]],
    max_examples: int = 10,
) -> tuple[int, list[dict[str, str]]]:
    adjacency = adjacency_from_edges(positive_edges)
    positive_lookup = {frozenset(edge) for edge in positive_edges}
    violation_count = 0
    examples: list[dict[str, str]] = []

    for middle in entity_ids:
        neighbors = sorted(adjacency.get(middle, set()))
        for left_idx, left in enumerate(neighbors):
            for right in neighbors[left_idx + 1 :]:
                if frozenset((left, right)) not in positive_lookup:
                    violation_count += 1
                    if len(examples) < max_examples:
                        examples.append(
                            {
                                "entity_a": left,
                                "entity_b": middle,
                                "entity_c": right,
                            }
                        )
    return violation_count, examples


def build_labeled_pairs(
    *,
    entity_ids: list[str],
    positive_edges: set[tuple[str, str]],
    negative_ratio: float,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    if negative_ratio < 0.0:
        raise ValueError("--negative-ratio must be non-negative.")

    positive_lookup = {frozenset(edge) for edge in positive_edges}
    positive_pairs: list[tuple[str, str, int]] = []
    negative_pairs: list[tuple[str, str, int]] = []

    for left, right in all_unordered_pairs(entity_ids):
        if frozenset((left, right)) in positive_lookup:
            positive_pairs.append((left, right, 1))
        else:
            negative_pairs.append((left, right, 0))

    requested_negative_count = int(round(len(positive_pairs) * negative_ratio))
    negative_count = min(requested_negative_count, len(negative_pairs))
    sampled_negatives = rng.sample(negative_pairs, negative_count) if negative_count else []

    pairs = positive_pairs + sampled_negatives
    if len({label for _, _, label in pairs}) < 2:
        raise ValueError("Selected pairs must contain both labels.")
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
    validation_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))

    atom_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for pair_idx, (left, right, label) in enumerate(unordered_pairs, start=1):
        zero_based_pair_idx = pair_idx - 1
        if rng.random() < 0.5:
            train_left, train_right = left, right
            reverse_left, reverse_right = right, left
        else:
            train_left, train_right = right, left
            reverse_left, reverse_right = left, right

        pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
        reverse_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in validation_indices else "test"
        pair_id = f"pair_{pair_idx:06d}"

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
                "entity_1": reverse_left,
                "entity_2": reverse_right,
                "label": label,
                "split": reverse_split,
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
                "reverse_entity_1": reverse_left,
                "reverse_entity_2": reverse_right,
                "reverse_split": reverse_split,
            }
        )

    return atom_rows, pair_rows


def validate_dataset(
    *,
    atom_rows: list[dict[str, object]],
    entity_ids: list[str],
    positive_edges: set[tuple[str, str]],
) -> dict[str, object]:
    required_columns = {"entity_1", "entity_2", "label", "split"}
    actual_columns = set(atom_rows[0]) if atom_rows else set()
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

    transitivity_violations, transitivity_violation_examples = count_transitivity_violations(
        entity_ids=entity_ids,
        positive_edges=positive_edges,
    )
    return {
        "has_required_columns": required_columns <= actual_columns,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": dict(sorted(split_label_counts.items())),
        "missing_reverse_count": len(missing_reverse_examples),
        "reverse_label_mismatch_count": len(reverse_label_mismatch_examples),
        "transitivity_violation_count": transitivity_violations,
        "transitivity_violation_examples": transitivity_violation_examples,
        "missing_reverse_examples": missing_reverse_examples[:10],
        "reverse_label_mismatch_examples": reverse_label_mismatch_examples[:10],
        "is_valid_symmetric_nontransitive": (
            required_columns <= actual_columns
            and len(missing_reverse_examples) == 0
            and len(reverse_label_mismatch_examples) == 0
            and transitivity_violations > 0
            and split_counts.get("train", 0) > 0
            and split_counts.get("test", 0) > 0
        ),
    }


def build_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation"]
    lines = [
        "# Non-Transitive Symmetry Dataset",
        "",
        "The model-facing table is `nontransitive_symmetry_atoms.csv`.",
        "",
        "The relation is symmetric because every unordered pair is exported in both directions with the same label.",
        "The relation is non-transitive because positive edges are sampled as an undirected graph, not as hidden groups.",
        "",
        "Use only `entity_1` and `entity_2` as features and `label` as the target.",
        "`split` is metadata for train/validation/test selection, not a model feature.",
        "",
        "## Shape",
        "",
        f"- entities: `{metadata['num_entities']}`",
        f"- positive unordered edges: `{metadata['num_positive_unordered_edges']}`",
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
    lines.append(f"- symmetric non-transitive dataset valid: `{validation['is_valid_symmetric_nontransitive']}`")
    lines.append(f"- reverse label mismatches: `{validation['reverse_label_mismatch_count']}`")
    lines.append(f"- transitivity violations: `{validation['transitivity_violation_count']}`")
    if validation["transitivity_violation_examples"]:
        lines.extend(["", "## Example Transitivity Violation", ""])
        example = validation["transitivity_violation_examples"][0]
        lines.append(
            "`R({a}, {b})=1` and `R({b}, {c})=1`, but `R({a}, {c})=0`.".format(
                a=example["entity_a"],
                b=example["entity_b"],
                c=example["entity_c"],
            )
        )
    return "\n".join(lines) + "\n"


def generate_positive_edges(
    *,
    entity_ids: list[str],
    positive_degree: int,
    max_generation_attempts: int,
    rng: random.Random,
) -> set[tuple[str, str]]:
    unordered_pairs = all_unordered_pairs(entity_ids)
    for _attempt in range(max_generation_attempts):
        positive_edges = sample_positive_edges(
            unordered_pairs=unordered_pairs,
            num_entities=len(entity_ids),
            positive_degree=positive_degree,
            rng=rng,
        )
        transitivity_violations, _examples = count_transitivity_violations(
            entity_ids=entity_ids,
            positive_edges=positive_edges,
        )
        if transitivity_violations > 0:
            return positive_edges
    raise RuntimeError(
        "Could not sample a non-transitive symmetric graph. Increase --num-entities, "
        "--positive-degree, or --max-generation-attempts."
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    entities = build_entities(args.num_entities, rng=rng)
    entity_ids = [str(row["entity_id"]) for row in entities]
    positive_edges = generate_positive_edges(
        entity_ids=entity_ids,
        positive_degree=args.positive_degree,
        max_generation_attempts=args.max_generation_attempts,
        rng=rng,
    )
    unordered_pairs = build_labeled_pairs(
        entity_ids=entity_ids,
        positive_edges=positive_edges,
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
    validation = validate_dataset(
        atom_rows=atom_rows,
        entity_ids=entity_ids,
        positive_edges=positive_edges,
    )
    pair_role_counts = Counter(str(row["pair_role"]) for row in pair_rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "num_entities": len(entities),
        "num_positive_unordered_edges": len(positive_edges),
        "num_unordered_pairs": len(unordered_pairs),
        "num_atom_rows": len(atom_rows),
        "positive_degree": args.positive_degree,
        "negative_ratio": args.negative_ratio,
        "validation_fraction": args.validation_fraction,
        "split_mode": args.split_mode,
        "demo_fraction": args.demo_fraction,
        "pair_role_counts": dict(sorted(pair_role_counts.items())),
        "entity_ids_randomized": True,
        "generation_rule": (
            "label is 1 for sampled undirected graph edges and 0 for sampled non-edges. "
            "Every selected unordered pair is exported in both directions with the same label. "
            "No hidden group structure is used, so positive edges do not force transitive closure."
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
        output_dir / "nontransitive_symmetry_atoms.csv",
        atom_rows,
        ["entity_1", "entity_2", "label", "split"],
    )
    write_csv(
        output_dir / "entities.csv",
        entities,
        ["entity_id", "internal_entity_id"],
    )
    write_csv(
        output_dir / "positive_edges.csv",
        [
            {
                "edge_id": f"edge_{idx:06d}",
                "entity_a": left,
                "entity_b": right,
            }
            for idx, (left, right) in enumerate(sorted(positive_edges), start=1)
        ],
        ["edge_id", "entity_a", "entity_b"],
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
            "reverse_entity_1",
            "reverse_entity_2",
            "reverse_split",
        ],
    )
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "validation_report.md").write_text(build_report(metadata), encoding="utf-8")

    if not validation["is_valid_symmetric_nontransitive"]:
        raise RuntimeError(f"Generated dataset failed validation. See {output_dir / 'metadata.json'}")

    print(f"Wrote non-transitive symmetry dataset to {output_dir}")
    print(
        "entities={entities} positive_edges={positive_edges} rows={rows} train={train} validation={validation} test={test}".format(
            entities=len(entities),
            positive_edges=len(positive_edges),
            rows=len(atom_rows),
            train=validation["split_counts"].get("train", 0),
            validation=validation["split_counts"].get("validation", 0),
            test=validation["split_counts"].get("test", 0),
        )
    )
    print(f"label_counts={validation['label_counts']}")
    print(f"transitivity_violations={validation['transitivity_violation_count']}")


if __name__ == "__main__":
    main()
