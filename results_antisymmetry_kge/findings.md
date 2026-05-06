# KGE Relation Baseline Findings

- data path: `minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.5208 | 0.5106 | 0.5906 | 0.5173 | 0.5000 | 35 |
| distmult | 0.3333 | 0.1795 | 0.4587 | 0.3247 | 0.5000 | 1 |
| transe | 0.9792 | 0.9808 | 0.9978 | 0.9969 | 0.5000 | 173 |
| transr | 0.9792 | 0.9808 | 0.9928 | 0.9873 | 0.5000 | 193 |
