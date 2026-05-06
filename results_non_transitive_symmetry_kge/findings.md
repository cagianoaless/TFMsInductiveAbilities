# KGE Relation Baseline Findings

- data path: `nontransitive_symmetry_dataset/nontransitive_symmetry_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.5234 | 0.4696 | 0.5481 | 0.5407 | 0.5000 | 1 |
| distmult | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 999 |
| transe | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1000 |
| transr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1000 |
