# Results Layout

Curated TFM/KGE result folders are grouped here by relation family and experimental protocol. No result files were deleted during the cleanup; the old root-level `results_*` folders were moved into this hierarchy.

Each leaf result folder normally contains:

```text
metrics.csv
predictions.csv
metadata.json
findings.md
```

## Symmetry

| New path | Former path | Notes |
| --- | --- | --- |
| `symmetry/hidden_group_random/tfms/` | `results_symmetry_tfms/` | Random-ID hidden-group symmetry, TabPFN/TabICL plus DistMult control. |
| `symmetry/hidden_group_random/kge/` | `results_symmetry_kge/` | Random-ID hidden-group symmetry, KGE baselines. |
| `symmetry/hidden_group_sequential/tfms/` | `results_symmetry_sequential_tfms/` | Sequential-ID hidden-group symmetry, TFM shortcut/leakage probe. |
| `symmetry/nontransitive/kge/` | `results_non_transitive_symmetry_kge/` | Non-transitive symmetric relation, KGE baselines. |
| `symmetry/demo_fraction/random_demo_0.0_tfms/` | `results_random_demo_0_0/` | Large random-ID symmetry, no demonstration pairs. |
| `symmetry/demo_fraction/random_demo_0.5_tfms/` | `results_random_demo_0_5/` | Large random-ID symmetry, 50% demonstration pairs. |
| `symmetry/demo_fraction/sequential_demo_0.0_tfms/` | `results_sequential_demo_0_0/` | Large sequential-ID symmetry, no demonstration pairs. |
| `symmetry/demo_fraction/sequential_demo_0.5_tfms/` | `results_sequential_demo_0_5/` | Large sequential-ID symmetry, 50% demonstration pairs. |

## Antisymmetry

| New path | Former path | Notes |
| --- | --- | --- |
| `antisymmetry/total_order/tfms/` | `results_antisymmetry_tfms/` | Total-order antisymmetry, TabPFN/TabICL plus DistMult control. |
| `antisymmetry/total_order/kge/` | `results_antisymmetry_kge/` | Total-order antisymmetry, KGE baselines. |
| `antisymmetry/supervised_by_sparse/tfms/` | `results_sparse_supervised_by_tfms/` | Sparse `supervised_by`, TFMs. |
| `antisymmetry/supervised_by_sparse/kge/` | `results_sparse_kge/` | Sparse `supervised_by`, KGE baselines. |
| `antisymmetry/supervised_by_balanced_small/tfms/` | `results_supervised_by_balanced_tfms/` | Balanced small `supervised_by`, TFMs. |
| `antisymmetry/supervised_by_balanced_small/kge/` | `results_supervised_by_balanced_kge/` | Balanced small `supervised_by`, KGE baselines. |
| `antisymmetry/supervised_by_balanced_big/tfms/` | `results_supervised_by_balanced_big_tfms/` | Balanced large `supervised_by`, TFMs. |
| `antisymmetry/supervised_by_balanced_big/kge/` | `results_supervised_by_balanced_big_kge/` | Balanced large `supervised_by`, KGE baselines. |
| `antisymmetry/supervised_by_balanced_big_no_demo/tfms/` | `results_supervised_by_balanced_big_nodemo/` | Balanced large `supervised_by`, no demonstration pairs, TFMs. |

## LLM Results

Ollama/LLM outputs remain in `llm_exp/` because the LLM scripts write and summarize results there:

```text
llm_exp/results/
llm_exp/prompt_sweep/
```
