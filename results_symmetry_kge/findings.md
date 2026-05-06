# KGE Relation Baseline Findings

- data path: `minimal_symmetry_dataset/symmetry_friendship_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.5000 | 0.5472 | 0.5394 | 0.5350 | 0.5000 | 41 |
| distmult | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 999 |
| transe | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1000 |
| transr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1000 |
