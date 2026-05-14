# Minimal Symmetry Dataset

The model-facing table is `symmetry_friendship_atoms.csv`.

Use only `entity_1` and `entity_2` as features and `label` as the target.
`split` is metadata for train/validation/test selection, not a model feature.

## Shape

- entities: `1000`
- hidden groups: `100`
- sampled unordered pairs: `9000`
- atom rows: `18000`
- split mode: `symmetry_demonstration`
- demo unordered pairs: `900`
- query unordered pairs: `8100`

## Split Counts

- `test`: `6480`
- `train`: `9900`
- `validation`: `1620`

## Validation

- symmetry dataset valid: `True`
- hidden group columns in model table: `[]`
- reverse label mismatches: `0`
