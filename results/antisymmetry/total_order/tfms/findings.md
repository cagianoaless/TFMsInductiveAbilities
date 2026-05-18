# Minimal Symmetry TFM Findings

- data path: `minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv`
- threshold mode: `fixed`
- device: `cuda`

| Model | Accuracy | F1 | Average Precision | ROC AUC | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| distmult | 0.3229 | 0.0000 | 0.3667 | 0.0851 | 0.5000 |
| tabicl | 0.6354 | 0.6535 | 0.7339 | 0.7280 | 0.5000 |
| tabpfn | 0.6250 | 0.6400 | 0.7413 | 0.7222 | 0.5000 |
