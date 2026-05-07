# Minimal Symmetry Dataset

The model-facing table is `symmetry_friendship_atoms.csv`.

Use only `entity_1` and `entity_2` as features and `label` as the target.
`split` is metadata for train/validation/test selection, not a model feature.

## Shape

- entities: `60`
- hidden groups: `12`
- sampled unordered pairs: `240`
- atom rows: `480`
- split mode: `symmetry_demonstration`
- demo unordered pairs: `2`
- query unordered pairs: `238`

## Split Counts

- `test`: `190`
- `train`: `242`
- `validation`: `48`

## Validation

- symmetry dataset valid: `True`
- hidden group columns in model table: `[]`
- reverse label mismatches: `0`
