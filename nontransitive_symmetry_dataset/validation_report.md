# Non-Transitive Symmetry Dataset

The model-facing table is `nontransitive_symmetry_atoms.csv`.

The relation is symmetric because every unordered pair is exported in both directions with the same label.
The relation is non-transitive because positive edges are sampled as an undirected graph, not as hidden groups.

Use only `entity_1` and `entity_2` as features and `label` as the target.
`split` is metadata for train/validation/test selection, not a model feature.

## Shape

- entities: `80`
- positive unordered edges: `160`
- sampled unordered pairs: `320`
- atom rows: `640`
- split mode: `reverse_holdout`
- demo unordered pairs: `0`
- query unordered pairs: `320`

## Split Counts

- `test`: `256`
- `train`: `320`
- `validation`: `64`

## Validation

- symmetric non-transitive dataset valid: `True`
- reverse label mismatches: `0`
- transitivity violations: `592`

## Example Transitivity Violation

`R(entity_0008, entity_0069)=1` and `R(entity_0069, entity_0010)=1`, but `R(entity_0008, entity_0010)=0`.
