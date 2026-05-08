# KGE Relation Baseline Findings

- data path: `balanced_supervised_by_dataset_big/supervised_by_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.7811 | 0.7883 | 0.8431 | 0.8610 | 0.5000 | 994 |
| distmult | 0.0000 | 0.0000 | 0.3230 | 0.0000 | 0.5000 | 774 |
| transe | 0.9966 | 0.9967 | 1.0000 | 1.0000 | 0.5000 | 1000 |
| transr | 0.8552 | 0.8377 | 1.0000 | 1.0000 | 0.5000 | 1000 |
