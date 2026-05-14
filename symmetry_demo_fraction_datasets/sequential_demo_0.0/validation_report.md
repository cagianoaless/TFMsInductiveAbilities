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
- demo unordered pairs: `0`
- query unordered pairs: `9000`

## Split Counts

- `test`: `7200`
- `train`: `9000`
- `validation`: `1800`

## Validation

- symmetry dataset valid: `True`
- hidden group columns in model table: `[]`
- reverse label mismatches: `0`
