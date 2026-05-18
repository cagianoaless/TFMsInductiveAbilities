# KGE Relation Baseline Findings

- data path: `sparse_supervised_by_dataset_balanced/supervised_by_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.7407 | 0.7586 | 0.8287 | 0.8297 | 0.5000 | 1000 |
| distmult | 0.1481 | 0.1481 | 0.3364 | 0.0989 | 0.5000 | 1 |
| transe | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1000 |
| transr | 0.5185 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 4 |
