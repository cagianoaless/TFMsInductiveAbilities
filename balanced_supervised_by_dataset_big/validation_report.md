# Sparse Supervised-By Dataset

The model-facing table is `supervised_by_atoms.csv`.

`label=1` means `entity_1` is supervised by `entity_2`.
The relation is sparse: only sampled student-supervisor edges are positive.
Unrelated pairs are negative in both directions, which is different from the old total-order antisymmetry dataset.

Use only `entity_1` and `entity_2` as features and `label` as the target.
`split` is metadata for train/validation/test selection, not a model feature.

## Shape

- students: `400`
- supervisors: `100`
- positive supervision edges: `594`
- sampled unrelated unordered pairs: `0`
- atom rows: `1188`
- positive rate: `0.5000`
- split mode: `supervision_demonstration`
- demo unordered pairs: `297`
- query unordered pairs: `297`

## Split Counts

- `test`: `297`
- `train`: `891`

## Validation

- supervised_by dataset valid: `True`
- sparse supervised_by dataset valid: `False`
- balanced supervised_by dataset valid: `True`
- balanced label counts: `True`
- both-positive antisymmetry violations: `0`
- positive rows whose reverse is negative: `594`
- both-negative unordered pairs: `0`
