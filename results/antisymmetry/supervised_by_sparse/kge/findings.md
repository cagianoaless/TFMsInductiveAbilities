# KGE Relation Baseline Findings

- data path: `sparse_supervised_by_dataset/supervised_by_atoms.csv`
- threshold mode: `tune_validation_f1`
- device: `cuda`
- embedding dim: `64`
- relation dim: `64`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complex | 0.6284 | 0.1980 | 0.2264 | 0.6160 | 0.4400 | 37 |
| distmult | 0.4174 | 0.1806 | 0.1105 | 0.4258 | 0.2300 | 27 |
| transe | 0.6101 | 0.3885 | 0.3253 | 0.8256 | 0.0860 | 20 |
| transr | 0.7661 | 0.2154 | 0.1941 | 0.6391 | 0.1600 | 8 |
