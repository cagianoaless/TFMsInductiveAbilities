#!/usr/bin/env python3
"""Print a compact inventory of experiment datasets, results, and scripts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_files(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def classify_dir(path: Path) -> str:
    name = path.name
    if name == "results" or name.startswith("results_"):
        return "result"
    if name in {"llm_exp", "docs", "tools"}:
        return "support"
    if name.endswith("_dataset") or "dataset" in name or name == "symmetry_demo_fraction_datasets":
        return "dataset"
    return "other"


def print_section(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(none)")
        return
    width = max(len(name) for name, _ in rows)
    for name, detail in rows:
        print(f"{name:<{width}}  {detail}")


def result_leaves(results_root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for metrics_path in sorted(results_root.glob("**/metrics.csv")):
        leaf = metrics_path.parent
        rows.append((str(leaf.relative_to(ROOT)), f"{count_files(leaf)} files"))
    return rows


def main() -> None:
    scripts: list[tuple[str, str]] = []
    docs: list[tuple[str, str]] = []
    datasets: list[tuple[str, str]] = []
    results: list[tuple[str, str]] = []
    support: list[tuple[str, str]] = []
    other: list[tuple[str, str]] = []

    for path in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith("."):
            continue
        if path.is_file():
            if path.suffix == ".py":
                scripts.append((path.name, f"{path.stat().st_size} bytes"))
            elif path.suffix in {".md", ".pdf"}:
                docs.append((path.name, f"{path.stat().st_size} bytes"))
            else:
                other.append((path.name, f"{path.stat().st_size} bytes"))
            continue

        if path.name == "results":
            results.extend(result_leaves(path))
            continue

        detail = f"{count_files(path)} files"
        kind = classify_dir(path)
        if kind == "dataset":
            datasets.append((path.name, detail))
        elif kind == "result":
            results.append((path.name, detail))
        elif kind == "support":
            support.append((path.name, detail))
        else:
            other.append((path.name, detail))

    print(f"Repository: {ROOT}")
    print_section("Scripts", scripts)
    print_section("Documentation", docs)
    print_section("Datasets", datasets)
    print_section("Results", results)
    print_section("Support", support)
    print_section("Other", other)


if __name__ == "__main__":
    main()
