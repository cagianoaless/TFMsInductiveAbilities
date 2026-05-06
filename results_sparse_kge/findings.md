# KGE Relation Baseline Findings

- data path: `sparse_supervised_by_dataset/supervised_by_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.6927 | 0.2118 | 0.2264 | 0.6160 | 0.5000 | 37 |
| distmult | 0.7569 | 0.0000 | 0.1105 | 0.4258 | 0.5000 | 27 |
| transe | 0.8716 | 0.0000 | 0.3253 | 0.8256 | 0.5000 | 20 |
| transr | 0.8716 | 0.0000 | 0.1941 | 0.6391 | 0.5000 | 8 |
