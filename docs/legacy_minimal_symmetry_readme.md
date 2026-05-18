# Minimal Symmetry Experiment

This folder contains a first-pass benchmark for testing whether tabular foundation models can exploit symmetry in a binary relation.

The model-facing table is intentionally minimal:

```text
entity_1,entity_2,label,split
```

Hidden group IDs are used only to generate and validate labels. They are written to `entities.csv` for audit, but they are not present in the model-facing atom table.

Exported entity IDs are randomized by default after hidden group assignment, so nearby entity numbers do not reveal group membership.

## Generate Data

```bash
python3 generate_minimal_symmetry_dataset.py \
  --output-dir minimal_symmetry_dataset \
  --num-groups 12 \
  --group-size 5 \
  --negative-ratio 1.0 \
  --split-mode symmetry_demonstration \
  --demo-fraction 0.5
```

In `symmetry_demonstration` mode, some pairs are explicit ICL examples: both `(A, B)` and `(B, A)` are in train with the same label. The remaining query pairs put one direction in train and the reverse direction in validation/test. This asks whether the model uses the demonstrated symmetry pattern on new pairs.

The older stricter mode is still available:

```bash
python3 generate_minimal_symmetry_dataset.py \
  --output-dir minimal_symmetry_dataset \
  --split-mode reverse_holdout
```

## Dry Run TFM Test

```bash
python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_symmetry_dataset/symmetry_friendship_atoms.csv \
  --dry-run
```

## Run TFMs

Pass real local checkpoints. The script does not auto-download or fake unavailable models.

```bash
TABPFN_DISABLE_TELEMETRY=1 python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_symmetry_dataset/symmetry_friendship_atoms.csv \
  --output-dir minimal_symmetry_results \
  --models tabpfn,tabicl \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda
```

Use only `entity_1` and `entity_2` as features. `label` is the target and `split` is metadata.

## Run A DistMult Baseline

The evaluator also supports a DistMult embedding baseline. DistMult has symmetry built into its score, so it is a useful positive-control model for the symmetry dataset:

```bash
python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_symmetry_dataset/symmetry_friendship_atoms.csv \
  --output-dir minimal_symmetry_distmult_results \
  --models distmult
```

For antisymmetry:

```bash
python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv \
  --output-dir minimal_antisymmetry_distmult_results \
  --models distmult
```

DistMult-specific controls:

```text
--distmult-embedding-dim 32
--distmult-epochs 1000
--distmult-lr 0.05
--distmult-weight-decay 1e-4
--distmult-batch-size 512
--distmult-patience 100
```

## Antisymmetry Dataset

Plain `coauthor` is symmetric, so the antisymmetric academic relation here is `senior_coauthor_of`: if `A` is senior to `B`, then `B` is not senior to `A`.

Generate it with:

```bash
python3 generate_minimal_antisymmetry_dataset.py \
  --output-dir minimal_antisymmetry_dataset \
  --num-entities 60 \
  --num-pairs 240 \
  --split-mode antisymmetry_demonstration \
  --demo-fraction 0.5
```

The same TFM runner can evaluate it because the model-facing schema is identical:

```bash
python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv \
  --dry-run
```

## Sparse `supervised_by` Dataset

The total-order antisymmetry dataset is useful, but it is easier than a realistic `supervised_by` relation because every pair has one positive direction. The sparse generator creates a directed graph instead: only student-to-supervisor edges are positive, reverse edges are negative, and unrelated pairs are negative in both directions.

```bash
python3 generate_sparse_supervised_by_dataset.py \
  --output-dir sparse_supervised_by_dataset \
  --num-students 48 \
  --num-supervisors 12 \
  --min-supervisors-per-student 1 \
  --max-supervisors-per-student 2 \
  --negative-ratio 3.0 \
  --split-mode reverse_holdout
```

Evaluate it with the same runner:

```bash
python3 run_minimal_symmetry_tfm_experiment.py \
  --data-path sparse_supervised_by_dataset/supervised_by_atoms.csv \
  --dry-run
```

KGE baselines use the same model-facing file:

```bash
python3 run_kge_relation_baselines.py \
  --data-path sparse_supervised_by_dataset/supervised_by_atoms.csv \
  --output-dir results_sparse_supervised_by_kge \
  --models complex,transr,transe,distmult \
  --device cuda
```
