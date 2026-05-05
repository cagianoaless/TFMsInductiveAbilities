# Minimal Antisymmetry Dataset

The model-facing table is `antisymmetry_coauthor_atoms.csv`.

Relation: `senior_coauthor_of`. This is directed and antisymmetric: if `A -> B = 1`, then `B -> A = 0`.

Use only `entity_1` and `entity_2` as features and `label` as the target.
`split` is metadata for train/validation/test selection, not a model feature.

## Shape

- entities: `60`
- sampled unordered pairs: `240`
- atom rows: `480`
- split mode: `antisymmetry_demonstration`
- demo unordered pairs: `120`
- query unordered pairs: `120`

## Split Counts

- `test`: `96`
- `train`: `360`
- `validation`: `24`

## Validation

- antisymmetry dataset valid: `True`
- hidden rank columns in model table: `[]`
- antisymmetry violations: `0`
