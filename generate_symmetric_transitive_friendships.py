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
            "Generate a table-based synthetic binary relation that is symmetric and transitive. "
            "The default example is fake friendship, implemented as fully connected friendship groups."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_friendship_relational"),
        help="Directory where CSV and metadata files will be written.",
    )
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="People per group when --group-sizes is not provided.",
    )
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="",
        help="Optional comma-separated group sizes, e.g. 4,6,8. Overrides --num-groups and --group-size.",
    )
    parser.add_argument(
        "--relation-name",
        type=str,
        default="fake_friendship",
        help="Name stored in the relation column.",
    )
    parser.add_argument(
        "--exclude-self-pairs",
        action="store_true",
        help=(
            "Omit rows where source_person_id == target_person_id. "
            "Use this only if you want friendship-like data; formal transitivity with symmetric positives "
            "requires positive self-pairs."
        ),
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Write only positive relation rows instead of the full labeled ordered-pair table.",
    )
    return parser.parse_args()


def parse_group_sizes(text: str, *, num_groups: int, group_size: int) -> list[int]:
    if text.strip():
        sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        sizes = [group_size for _ in range(num_groups)]

    if not sizes:
        raise ValueError("At least one group is required.")
    if any(size <= 0 for size in sizes):
        raise ValueError("All group sizes must be positive.")
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


def build_people(group_sizes: list[int], seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    first_names = [
        "Ada",
        "Bruno",
        "Carla",
        "Dario",
        "Elena",
        "Farah",
        "Giorgio",
        "Hana",
        "Iris",
        "Jonas",
        "Kira",
        "Luca",
        "Maya",
        "Noah",
        "Olivia",
        "Pavel",
        "Rina",
        "Sara",
        "Tomas",
        "Vera",
    ]

    people: list[dict[str, object]] = []
    person_index = 1
    for group_index, size in enumerate(group_sizes, start=1):
        group_id = f"group_{group_index:02d}"
        for local_index in range(1, size + 1):
            name = f"{rng.choice(first_names)} {group_index:02d}-{local_index:02d}"
            people.append(
                {
                    "person_id": f"person_{person_index:04d}",
                    "display_name": name,
                    "friendship_group_id": group_id,
                    "group_local_index": local_index,
                }
            )
            person_index += 1
    return people


def build_relation_tables(
    *,
    people: list[dict[str, object]],
    relation_name: str,
    exclude_self_pairs: bool,
    positive_only: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atoms: list[dict[str, object]] = []
    positive_rows: list[dict[str, object]] = []
    atom_index = 1
    positive_index = 1

    for source in people:
        for target in people:
            source_id = str(source["person_id"])
            target_id = str(target["person_id"])
            if exclude_self_pairs and source_id == target_id:
                continue

            same_group = source["friendship_group_id"] == target["friendship_group_id"]
            label = int(same_group)
            if positive_only and label == 0:
                continue

            atom = {
                "atom_id": f"atom_{atom_index:06d}",
                "relation": relation_name,
                "source_person_id": source_id,
                "target_person_id": target_id,
                "source_group_id": source["friendship_group_id"],
                "target_group_id": target["friendship_group_id"],
                "label": label,
            }
            atoms.append(atom)
            atom_index += 1

            if label == 1:
                positive_rows.append(
                    {
                        "friendship_id": f"friendship_{positive_index:06d}",
                        "source_person_id": source_id,
                        "target_person_id": target_id,
                        "relation": relation_name,
                    }
                )
                positive_index += 1

    return atoms, positive_rows


def validate_relation(
    atoms: list[dict[str, object]],
    *,
    exclude_self_pairs: bool,
    positive_only: bool,
) -> dict[str, object]:
    labels: dict[tuple[str, str], int] = {}
    people: set[str] = set()
    for row in atoms:
        source = str(row["source_person_id"])
        target = str(row["target_person_id"])
        labels[(source, target)] = int(row["label"])
        people.add(source)
        people.add(target)

    positive_pairs = {pair for pair, label in labels.items() if label == 1}
    negative_pairs = {pair for pair, label in labels.items() if label == 0}

    missing_symmetric_pairs: list[tuple[str, str]] = []
    for source, target in positive_pairs:
        if labels.get((target, source), 0) != 1:
            missing_symmetric_pairs.append((source, target))

    transitivity_violations: list[tuple[str, str, str]] = []
    positives_by_middle: dict[str, list[tuple[str, str]]] = defaultdict(list)
    positives_from_middle: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, target in positive_pairs:
        positives_by_middle[target].append((source, target))
        positives_from_middle[source].append((source, target))

    for middle in sorted(people):
        incoming = positives_by_middle.get(middle, [])
        outgoing = positives_from_middle.get(middle, [])
        for source, _ in incoming:
            for _, target in outgoing:
                if labels.get((source, target), 0) != 1:
                    transitivity_violations.append((source, middle, target))
                    if len(transitivity_violations) >= 20:
                        break
            if len(transitivity_violations) >= 20:
                break
        if len(transitivity_violations) >= 20:
            break

    formal_note = None
    if exclude_self_pairs:
        formal_note = (
            "Self-pairs were excluded. Symmetric off-diagonal positives cannot satisfy formal transitivity "
            "when paths like A->B and B->A imply A->A. Use the default self-pair setting for a formal "
            "symmetric and transitive relation."
        )
    if positive_only:
        formal_note = (
            "Only positive rows were written. Missing rows are not explicit negatives, so validation is "
            "performed over the observed positive table."
        )

    return {
        "num_people_seen": len(people),
        "num_rows": len(atoms),
        "num_positive_rows": len(positive_pairs),
        "num_negative_rows": len(negative_pairs),
        "is_symmetric": len(missing_symmetric_pairs) == 0,
        "is_transitive": len(transitivity_violations) == 0,
        "missing_symmetric_pair_examples": missing_symmetric_pairs[:20],
        "transitivity_violation_examples": transitivity_violations[:20],
        "formal_note": formal_note,
    }


def build_validation_report(metadata: dict[str, object]) -> str:
    validation = metadata["validation"]
    label_counts = metadata["label_counts"]
    lines = [
        "# Synthetic Friendship Validation Report",
        "",
        "## Dataset Shape",
        "",
        f"- people: `{metadata['num_people']}`",
        f"- groups: `{metadata['num_groups']}`",
        f"- relation rows: `{metadata['num_relation_rows']}`",
        f"- positive rows: `{label_counts.get('1', 0)}`",
        f"- negative rows: `{label_counts.get('0', 0)}`",
        "",
        "## Relation Properties",
        "",
        f"- symmetric: `{validation['is_symmetric']}`",
        f"- transitive: `{validation['is_transitive']}`",
    ]
    if validation["formal_note"]:
        lines.extend(["", "## Note", "", str(validation["formal_note"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    group_sizes = parse_group_sizes(args.group_sizes, num_groups=args.num_groups, group_size=args.group_size)
    people = build_people(group_sizes, args.seed)
    atoms, positive_rows = build_relation_tables(
        people=people,
        relation_name=args.relation_name,
        exclude_self_pairs=args.exclude_self_pairs,
        positive_only=args.positive_only,
    )
    validation = validate_relation(
        atoms,
        exclude_self_pairs=args.exclude_self_pairs,
        positive_only=args.positive_only,
    )

    label_counts = Counter(str(row["label"]) for row in atoms)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "people.csv",
        people,
        ["person_id", "display_name", "friendship_group_id", "group_local_index"],
    )
    write_csv(
        output_dir / "friendships.csv",
        positive_rows,
        ["friendship_id", "source_person_id", "target_person_id", "relation"],
    )
    write_csv(
        output_dir / "friendship_relation_atoms.csv",
        atoms,
        [
            "atom_id",
            "relation",
            "source_person_id",
            "target_person_id",
            "source_group_id",
            "target_group_id",
            "label",
        ],
    )
    write_csv(
        output_dir / "relation_types.csv",
        [
            {
                "relation_id": "R01",
                "relation_name": args.relation_name,
                "arity": 2,
                "is_symmetric": 1,
                "is_transitive": 1,
                "is_reflexive": int(not args.exclude_self_pairs),
                "description": "Synthetic group-based friendship relation.",
            }
        ],
        ["relation_id", "relation_name", "arity", "is_symmetric", "is_transitive", "is_reflexive", "description"],
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "relation_name": args.relation_name,
        "num_groups": len(group_sizes),
        "group_sizes": group_sizes,
        "num_people": len(people),
        "num_relation_rows": len(atoms),
        "num_positive_friendship_rows": len(positive_rows),
        "exclude_self_pairs": bool(args.exclude_self_pairs),
        "positive_only": bool(args.positive_only),
        "label_counts": dict(sorted(label_counts.items())),
        "generation_rule": (
            "Two people are positive examples iff they belong to the same friendship group. "
            "Each group is therefore a clique/equivalence class."
        ),
        "validation": validation,
    }
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "validation_report.md").write_text(build_validation_report(metadata), encoding="utf-8")

    if not validation["is_symmetric"] or not validation["is_transitive"]:
        raise RuntimeError(f"Generated relation failed validation. See {output_dir / 'metadata.json'}")

    print(f"Wrote synthetic friendship tables to {output_dir}")
    print(f"people={len(people)} rows={len(atoms)} positives={label_counts.get('1', 0)} negatives={label_counts.get('0', 0)}")
    print(f"symmetric={validation['is_symmetric']} transitive={validation['is_transitive']}")


if __name__ == "__main__":
    main()
