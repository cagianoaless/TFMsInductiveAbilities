# Repository Map

This file is the navigation layer for the experiment repository. It documents the current folders without moving or deleting any existing artifacts.

## Root Files

| Path | Purpose |
| --- | --- |
| `README.md` | High-level entry point and most common commands. |
| `NOTES.md` | Long-form working notes and recovered experiment history. |
| `experiments_report.md` | Academic-style report of the main experiments and interpretations. |
| `code_study_guide.md` | Code-oriented explanation of the generators and runners. |
| `run_minimal_symmetry_tfm_experiment_walkthrough.md` | Detailed walkthrough of the TFM runner and dataset logic. |

PDF files are exported copies of the corresponding Markdown reports.

## Dataset Generators

| Script | Generates |
| --- | --- |
| `generate_minimal_symmetry_dataset.py` | Hidden-group symmetric friendship-style relation. |
| `generate_nontransitive_symmetry_dataset.py` | Symmetric relation from sparse undirected edges, with transitivity violations. |
| `generate_minimal_antisymmetry_dataset.py` | Total-order antisymmetric seniority relation. |
| `generate_sparse_supervised_by_dataset.py` | Sparse directed `supervised_by` relation. |
| `generate_symmetric_transitive_friendships.py` | Earlier simple symmetric/transitive table generator. |

## Model Runners

| Script | Models |
| --- | --- |
| `run_minimal_symmetry_tfm_experiment.py` | TabPFN, TabICL, DistMult through the common atom-table schema. |
| `run_kge_relation_baselines.py` | ComplEx, TransR, TransE, DistMult KGE baselines. |
| `llm_exp/run_ollama_symmetry_experiment.py` | Local/cloud Ollama LLMs with sampled ICL batches. |
| `llm_exp/run_prompt_sweep.sh` | Prompt-template sweep for Ollama models. |

## Dataset Folders

| Folder | Meaning |
| --- | --- |
| `minimal_symmetry_dataset/` | Original random-ID hidden-group symmetry dataset. |
| `minimal_symmetry_dataset_sequential/` | Sequential-ID version, useful as an ID-leakage probe. |
| `symmetry_demo_fraction_datasets/` | Large random/sequential symmetry datasets across demo fractions. |
| `nontransitive_symmetry_dataset/` | Symmetric but intentionally non-transitive relation. |
| `minimal_antisymmetry_dataset/` | Total-order antisymmetry dataset. |
| `balanced_supervised_by_dataset*/` | Balanced sparse directed `supervised_by` datasets. |

Each dataset folder generally contains:

```text
*_atoms.csv
entities.csv
metadata.json
pair_assignments.csv
validation_report.md
```

Some sparse graph datasets also contain `positive_edges.csv` or `positive_supervisions.csv`.

## Result Folders

| Folder pattern | Meaning |
| --- | --- |
| `results/symmetry/` | Symmetry TFM/KGE outputs. |
| `results/antisymmetry/` | Total-order and `supervised_by` antisymmetry TFM/KGE outputs. |
| `llm_exp/results/` | Ollama LLM outputs for one prompt configuration. |
| `llm_exp/prompt_sweep/` | Ollama prompt comparison outputs. |

Standard result folders contain:

```text
metrics.csv
predictions.csv
metadata.json
findings.md
```

LLM result folders can also contain `per_run_metrics.csv`, `raw_responses.jsonl`, and `invalid_responses.jsonl`.

## Recommended Reading Order

1. `README.md`
2. `docs/REPO_MAP.md`
3. `experiments_report.md`
4. `code_study_guide.md`
5. `run_minimal_symmetry_tfm_experiment_walkthrough.md`

Use `tools/repo_inventory.py` when the repo grows again:

```bash
python3 tools/repo_inventory.py
```

The old root-level `results_*` folders were moved under `results/` without deleting files. See `results/README.md` for the exact mapping.
