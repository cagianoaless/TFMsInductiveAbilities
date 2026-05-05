# Code Walkthrough: Minimal Relational Ability Experiments

This document explains the Python code used in `experiment_2` at a level close to commented source code.

It covers three scripts:

- `generate_minimal_symmetry_dataset.py`
- `generate_minimal_antisymmetry_dataset.py`
- `run_minimal_symmetry_tfm_experiment.py`

The purpose is to make the experiment auditable: you should be able to read this file, open the scripts, and understand exactly how rows are generated, how train/test splits are created, how models receive data, and how predictions are evaluated.

## 1. High-Level Idea

The experiment tests whether tabular foundation models can learn relational properties from a table of ordered entity pairs.

The model-facing table always has this schema:

```text
entity_1,entity_2,label,split
```

The runner script is intentionally generic. It does not know whether the dataset is symmetry or antisymmetry. It only knows that:

- `entity_1` and `entity_2` are the input features;
- `label` is the binary target;
- `split` says whether a row is train, validation, or test.

The relational rule is created by the dataset generator.

For symmetry:

```text
label(A, B) = label(B, A)
```

For antisymmetry:

```text
label(A, B) = 1 - label(B, A)
```

This separation matters. If you want to audit the scientific validity of the experiment, most of the important logic is in the dataset generators, not in the model runner.

## 2. Shared Model-Facing Contract

Both generated datasets export a CSV that the models can read.

Symmetry output:

```text
minimal_symmetry_dataset/symmetry_friendship_atoms.csv
```

Antisymmetry output:

```text
minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv
```

Both contain only:

```text
entity_1,entity_2,label,split
```

This is the core anti-leakage rule.

Hidden variables are written only to audit files, for example:

- `entities.csv`
- `pair_assignments.csv`
- `metadata.json`
- `validation_report.md`

The runner rejects obvious hidden columns if they appear in the model-facing table. For example, it rejects:

```text
hidden_group_id
source_group_id
target_group_id
group_id
```

The symmetry generator also treats these as forbidden:

```text
group_id
hidden_group_id
entity_1_group
entity_2_group
```

The antisymmetry generator also treats these as forbidden:

```text
hidden_seniority_rank
rank
seniority
internal_entity_id
```

So the model is not given the hidden reason why a label is true or false. It only sees symbolic entity IDs.

## 3. Execution Flow

The usual workflow is:

```text
generator script
    creates entity metadata
    creates unordered pairs
    assigns labels
    assigns train/validation/test split
    writes model-facing CSV plus audit files

runner script
    reads model-facing CSV
    validates columns and labels
    builds categorical features from entity_1/entity_2
    fits selected models on train rows
    optionally tunes threshold on validation rows
    evaluates on test rows
    writes metrics and predictions
```

The generator decides what is being tested.

The runner decides how models are fit and evaluated.

## 4. Symmetry Generator: File Purpose

File:

```text
generate_minimal_symmetry_dataset.py
```

This script generates a fake friendship-like relation.

The hidden world is made of groups. Entities inside the same hidden group are considered friends. Entities in different hidden groups are not friends.

The model never sees the hidden group.

The rule is:

```text
label = 1 if entity_1 and entity_2 belong to the same hidden group
label = 0 otherwise
```

Because group membership does not depend on order:

```text
label(A, B) = label(B, A)
```

So the generated relation is symmetric.

## 5. Symmetry Generator: Imports

The script imports only standard-library modules:

```python
import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
```

Meaning:

- no NumPy;
- no pandas;
- no model dependency;
- no GPU dependency.

The generator is deliberately lightweight and deterministic.

`random.Random(args.seed)` is used instead of global random state, so the output is reproducible from the seed.

## 6. Symmetry Generator: `parse_args()`

`parse_args()` defines the command-line interface.

Important arguments:

- `--output-dir`: where the dataset files are written.
- `--seed`: controls reproducibility.
- `--num-groups`: number of hidden friendship groups.
- `--group-size`: number of entities per group.
- `--group-sizes`: optional comma-separated custom group sizes.
- `--negative-ratio`: how many negative unordered pairs to sample relative to positive unordered pairs.
- `--validation-fraction`: fraction of query reverse rows assigned to validation instead of test.
- `--split-mode`: either `reverse_holdout` or `symmetry_demonstration`.
- `--demo-fraction`: fraction of unordered pairs used as explicit train-only demonstrations.
- `--sequential-entity-ids`: debug option that disables entity ID randomization.

The most important methodological arguments are `--split-mode` and `--demo-fraction`.

`reverse_holdout` means:

```text
(A, B) goes to train
(B, A) goes to validation or test
```

for every selected pair.

`symmetry_demonstration` means some pairs become explicit demonstrations:

```text
(A, B) goes to train
(B, A) also goes to train
```

and the remaining query pairs follow reverse holdout:

```text
(A, B) goes to train
(B, A) goes to validation or test
```

This is the mechanism that adds in-context examples of symmetry.

## 7. Symmetry Generator: `parse_group_sizes()`

Function:

```python
def parse_group_sizes(text: str, *, num_groups: int, group_size: int) -> list[int]:
```

This function decides the hidden group sizes.

If `--group-sizes` is provided, for example:

```text
--group-sizes 4,5,6
```

the script uses exactly those sizes.

If `--group-sizes` is empty, the script creates:

```python
[group_size for _ in range(num_groups)]
```

So with the default:

```text
--num-groups 12
--group-size 5
```

the hidden world has:

```text
12 groups x 5 entities = 60 entities
```

The function validates two things:

- at least two groups must exist, otherwise there are no between-group negative examples;
- every group must contain at least two entities, otherwise there are no within-group positive pairs.

This is important because the task needs both labels.

## 8. Symmetry Generator: `write_csv()` And `write_json()`

These helper functions are small but important.

`write_csv()`:

```python
path.parent.mkdir(parents=True, exist_ok=True)
```

creates the output directory if needed.

Then it writes a header and all rows with `csv.DictWriter`.

`write_json()` does the same for JSON metadata.

These helpers ensure every output file is written in a consistent format.

## 9. Symmetry Generator: `build_entities()`

Function:

```python
def build_entities(
    group_sizes: list[int],
    *,
    rng: random.Random,
    sequential_entity_ids: bool,
) -> list[dict[str, object]]:
```

This creates the synthetic entities.

The function first creates internal records:

```text
internal_entity_id
hidden_group_id
group_local_index
```

Example internal record:

```text
internal_entity_0001, group_001, 1
```

The hidden group is the true reason why labels are generated.

Then the function creates exported entity IDs:

```python
exported_ids = [f"entity_{idx:04d}" for idx in range(1, len(internal_records) + 1)]
```

By default, these exported IDs are shuffled:

```python
if not sequential_entity_ids:
    rng.shuffle(exported_ids)
```

This is very important.

If IDs were not shuffled, entities in the same group could have adjacent numbers. A model might exploit entity ID order instead of learning the relation.

Example of dangerous non-randomized IDs:

```text
entity_0001, entity_0002, entity_0003, entity_0004, entity_0005 are all group_001
```

A model could learn a numeric shortcut.

With randomized IDs, group membership is not visible from the entity number.

The returned `entities` list still contains hidden group metadata, but that list is written only to `entities.csv`, not to the model-facing atom table.

## 10. Symmetry Generator: `build_unordered_pairs()`

Function:

```python
def build_unordered_pairs(
    entities: list[dict[str, object]],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
```

This function creates all possible unordered entity pairs.

The nested loop is:

```python
for left_idx, left in enumerate(entities):
    for right in entities[left_idx + 1 :]:
```

The slice `entities[left_idx + 1 :]` is the key detail.

It means the function creates only one unordered pair for each pair of entities.

It creates:

```text
(A, B)
```

but not:

```text
(B, A)
```

at this stage.

The reverse direction is added later by `assign_symmetry_split()`.

The label is computed as:

```python
label = int(left["hidden_group_id"] == right["hidden_group_id"])
```

So:

- same hidden group gives `1`;
- different hidden group gives `0`.

The function returns two lists:

- `positive_pairs`;
- `negative_pairs`.

Keeping positives and negatives separate allows controlled negative sampling.

## 11. Symmetry Generator: Why Negative Sampling Exists

In a grouped world, negative pairs are much more common than positive pairs.

With 12 groups of 5 entities:

- positive unordered pairs per group: `5 choose 2 = 10`;
- total positives: `12 x 10 = 120`;
- total unordered pairs among 60 entities: `60 choose 2 = 1770`;
- negatives: `1770 - 120 = 1650`.

Without sampling, the dataset would be very imbalanced:

```text
120 positive pairs vs 1650 negative pairs
```

That would make the task partly a class-imbalance benchmark.

The `--negative-ratio` argument controls this.

Default:

```text
--negative-ratio 1.0
```

means:

```text
sample about one negative unordered pair for each positive unordered pair
```

So the default selected unordered pairs are approximately balanced:

```text
120 positives + 120 sampled negatives
```

This is why the current generated symmetry dataset has 240 unordered pairs and 480 directed atom rows before split accounting.

## 12. Symmetry Generator: `sample_pairs()`

Function:

```python
def sample_pairs(
    *,
    positive_pairs: list[tuple[str, str, int]],
    negative_pairs: list[tuple[str, str, int]],
    negative_ratio: float,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
```

This function chooses which unordered pairs will be used.

It first checks:

```python
if negative_ratio < 0.0:
    raise ValueError("--negative-ratio must be non-negative.")
```

Then it computes:

```python
requested_negative_count = int(round(len(positive_pairs) * negative_ratio))
```

If there are 120 positive pairs and `negative_ratio=1.0`, it requests 120 negatives.

It never samples more negatives than available:

```python
negative_count = min(requested_negative_count, len(negative_pairs))
```

Then it samples negative pairs randomly:

```python
sampled_negatives = rng.sample(negative_pairs, negative_count)
```

Finally it combines all positives and sampled negatives:

```python
pairs = list(positive_pairs) + sampled_negatives
rng.shuffle(pairs)
```

All positive pairs are kept. Negatives are sampled.

That means the default symmetry dataset is not the full relation matrix. It is a sampled relation table.

## 13. Symmetry Generator: `choose_demo_indices()`

Function:

```python
def choose_demo_indices(
    unordered_pairs: list[tuple[str, str, int]],
    *,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> set[int]:
```

This function decides which unordered pairs are used as explicit demonstrations.

If the split mode is:

```text
reverse_holdout
```

it returns an empty set:

```python
return set()
```

This means no pair has both directions in train.

If the split mode is:

```text
symmetry_demonstration
```

the function samples demo pairs separately for each label.

It builds:

```python
indices_by_label: dict[int, list[int]] = {0: [], 1: []}
```

Then it appends each unordered pair index to label `0` or label `1`.

This is important because it avoids choosing demos only from one class. The model receives demonstrations for both positive and negative examples when possible.

For each label, it computes:

```python
demo_count = int(round(len(indices) * demo_fraction))
demo_count = min(max(demo_count, 1), len(indices) - 1)
```

This logic means:

- at least one demo is selected if that class has enough examples;
- at least one query pair remains for that class;
- demo fraction must be less than `1.0`.

So `demo_fraction=0.5` means roughly half of the positive unordered pairs and half of the negative unordered pairs are explicit train-only demonstrations.

## 14. Symmetry Generator: `assign_symmetry_split()`

Function:

```python
def assign_symmetry_split(
    *,
    unordered_pairs: list[tuple[str, str, int]],
    validation_fraction: float,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
```

This is the most important function in the symmetry generator.

It takes unordered pairs and turns each one into two directed rows:

```text
(A, B)
(B, A)
```

For symmetry, both rows receive the same label.

The function first chooses demo indices:

```python
demo_indices = choose_demo_indices(...)
```

Then it defines query indices:

```python
query_indices = [idx for idx in range(len(unordered_pairs)) if idx not in demo_indices]
```

Query pairs are the pairs whose reverse direction will be evaluated.

If all pairs were demos, there would be no test task, so the function rejects that:

```python
if not query_indices:
    raise ValueError("No query pairs remain after selecting demonstration pairs.")
```

Then it chooses which query reverse rows go to validation:

```python
reverse_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))
```

Query pairs not selected for validation become test rows.

Inside the main loop, the code randomly decides which direction is train:

```python
if rng.random() < 0.5:
    train_left, train_right = left, right
    holdout_left, holdout_right = right, left
else:
    train_left, train_right = right, left
    holdout_left, holdout_right = left, right
```

This avoids always training on the same arbitrary direction.

Then the code assigns the pair role:

```python
pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
```

The most important split line is:

```python
holdout_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in reverse_indices else "test"
```

Read it as:

- if the pair is a demo, the reverse direction also goes to train;
- else, if the pair was sampled for validation, the reverse direction goes to validation;
- else, the reverse direction goes to test.

Then it writes two model-facing rows.

First row:

```python
{
    "entity_1": train_left,
    "entity_2": train_right,
    "label": label,
    "split": "train",
}
```

Second row:

```python
{
    "entity_1": holdout_left,
    "entity_2": holdout_right,
    "label": label,
    "split": holdout_split,
}
```

So every unordered pair becomes two directed atom rows.

For demo pairs:

```text
(A, B), label, train
(B, A), label, train
```

For query pairs:

```text
(A, B), label, train
(B, A), label, validation/test
```

The function also writes `pair_rows`.

`pair_rows` are not used by the model. They are audit rows that preserve:

- `pair_id`;
- `pair_role`;
- unordered pair identity;
- train direction;
- reverse direction;
- reverse split.

This is where `pair_role` exists. It does not appear in the model-facing CSV.

## 15. Symmetry Generator: `validate_dataset()`

Function:

```python
def validate_dataset(atom_rows: list[dict[str, object]]) -> dict[str, object]:
```

This function checks whether the generated atom table is valid.

It checks required columns:

```python
required_columns = {"entity_1", "entity_2", "label", "split"}
```

It checks forbidden model-facing columns:

```python
forbidden_feature_columns = {"group_id", "hidden_group_id", "entity_1_group", "entity_2_group"}
```

Then it builds:

```python
labels_by_direction[(entity_1, entity_2)] = label
```

This allows the validator to check every directed row against its reverse.

For every `(entity_1, entity_2)`, it searches for `(entity_2, entity_1)`.

If the reverse is missing, that is a structural error.

If the reverse has a different label, that violates symmetry.

The relevant check is:

```python
elif reverse_label != label:
    reverse_label_mismatch_examples.append((entity_1, entity_2))
```

The dataset is considered valid only if:

- required columns exist;
- forbidden columns are absent;
- every row has a reverse row;
- reverse labels match;
- train split is non-empty;
- test split is non-empty.

The function returns counts and examples, not just a boolean. That makes debugging easier if generation fails.

## 16. Symmetry Generator: `build_report()`

`build_report(metadata)` creates a short human-readable Markdown report.

It includes:

- number of entities;
- number of hidden groups;
- number of sampled unordered pairs;
- number of atom rows;
- split mode;
- demo/query counts;
- split counts;
- validation status.

This file is useful for quickly checking a generated dataset without opening the JSON metadata.

## 17. Symmetry Generator: `main()`

`main()` is the orchestration function.

The order is:

```text
parse arguments
parse group sizes
create seeded random generator
build entities
build positive and negative unordered pairs
sample negatives according to negative_ratio
assign train/validation/test split
validate dataset
assemble metadata
write CSV, JSON, and Markdown outputs
raise error if validation fails
print summary
```

The model-facing file written by this script is:

```text
symmetry_friendship_atoms.csv
```

The audit files are:

```text
entities.csv
pair_assignments.csv
metadata.json
validation_report.md
```

The important scientific guarantee is:

```text
hidden_group_id is used to generate labels, but it is not written to symmetry_friendship_atoms.csv.
```

## 18. Antisymmetry Generator: File Purpose

File:

```text
generate_minimal_antisymmetry_dataset.py
```

This script generates a directed relation called:

```text
senior_coauthor_of
```

The name matters. Plain `coauthor_of` would normally be symmetric: if A coauthored with B, then B coauthored with A.

To test antisymmetry, the relation must be directional.

The intended meaning is:

```text
entity_1 is senior to entity_2
```

So if:

```text
senior_coauthor_of(A, B) = 1
```

then:

```text
senior_coauthor_of(B, A) = 0
```

This is antisymmetry in the binary-label setting used here.

## 19. Antisymmetry Generator: Differences From Symmetry

The antisymmetry generator is structurally similar to the symmetry generator.

The main differences are:

- there are no hidden groups;
- each entity has a hidden seniority rank;
- labels are directional;
- reverse labels must be opposite, not equal;
- there is no `negative_ratio` because every sampled unordered pair naturally produces one positive direction and one negative direction.

The model-facing output is:

```text
antisymmetry_coauthor_atoms.csv
```

It still contains only:

```text
entity_1,entity_2,label,split
```

## 20. Antisymmetry Generator: `parse_args()`

Important arguments:

- `--output-dir`: output folder.
- `--seed`: reproducibility seed.
- `--num-entities`: number of synthetic authors.
- `--num-pairs`: number of unordered author pairs to sample.
- `--validation-fraction`: fraction of query reverse rows assigned to validation.
- `--split-mode`: `reverse_holdout` or `antisymmetry_demonstration`.
- `--demo-fraction`: fraction of unordered pairs used as explicit demos.
- `--sequential-entity-ids`: debug option that disables random ID assignment.

The default is:

```text
--num-entities 60
--num-pairs 240
--split-mode antisymmetry_demonstration
--demo-fraction 0.5
```

This creates 240 unordered pairs.

Each unordered pair becomes two directed atom rows, so the model-facing table has 480 rows.

## 21. Antisymmetry Generator: `build_authors()`

Function:

```python
def build_authors(
    *,
    num_entities: int,
    rng: random.Random,
    sequential_entity_ids: bool,
) -> list[dict[str, object]]:
```

This function creates synthetic authors.

Each author gets an internal hidden seniority rank:

```python
"hidden_seniority_rank": idx
```

Rank `1` is the most senior under the current code, because later the label rule checks:

```python
left_rank < right_rank
```

The function creates exported IDs:

```python
entity_0001
entity_0002
...
```

By default, it shuffles them:

```python
if not sequential_entity_ids:
    rng.shuffle(exported_ids)
```

This prevents an easy shortcut where lower entity ID means more senior.

The audit file `entities.csv` contains:

```text
entity_id,internal_entity_id,hidden_seniority_rank
```

But the model-facing atom table does not contain `hidden_seniority_rank`.

So the model cannot directly read seniority.

## 22. Antisymmetry Generator: `build_unordered_pairs()`

Function:

```python
def build_unordered_pairs(authors: list[dict[str, object]]) -> list[tuple[str, str, int, int]]:
```

This creates all possible unordered pairs of authors.

Again, the loop uses:

```python
for right in authors[left_idx + 1 :]:
```

So each pair appears once at this stage.

Each returned tuple contains:

```text
left_entity_id
right_entity_id
left_hidden_rank
right_hidden_rank
```

The hidden ranks are kept here because the generator needs them to create labels.

They are not written to the model-facing CSV.

## 23. Antisymmetry Generator: `sample_pairs()`

Function:

```python
def sample_pairs(
    *,
    pairs: list[tuple[str, str, int, int]],
    num_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str, int, int]]:
```

This samples a fixed number of unordered author pairs.

If `--num-pairs 240`, then 240 unordered pairs are selected.

The sample count is capped at the number of available pairs:

```python
sample_count = min(num_pairs, len(pairs))
```

The sampled list is shuffled.

Unlike symmetry, there is no negative sampling here.

Why?

Because every unordered pair generates:

```text
one positive direction
one negative direction
```

So the final directed atom table is naturally balanced.

## 24. Antisymmetry Generator: `choose_demo_indices()`

This function is simpler than the symmetry version.

It does not stratify by label because every unordered pair already contains both labels after directions are created.

If split mode is:

```text
reverse_holdout
```

it returns an empty set.

If split mode is:

```text
antisymmetry_demonstration
```

it samples:

```python
demo_count = int(round(len(sampled_pairs) * demo_fraction))
```

and clamps it so at least one query pair remains:

```python
demo_count = min(max(demo_count, 1), len(sampled_pairs) - 1)
```

Demo pairs will have both directions in train.

Query pairs will have one direction in train and the reverse direction in validation/test.

## 25. Antisymmetry Generator: `directed_labels()`

Function:

```python
def directed_labels(
    left: str,
    right: str,
    left_rank: int,
    right_rank: int,
) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
```

This function converts one unordered pair into two directed labeled rows.

If `left_rank < right_rank`, then `left` is more senior:

```python
return (left, right, 1), (right, left, 0)
```

Otherwise, `right` is more senior:

```python
return (right, left, 1), (left, right, 0)
```

So the function always returns:

```text
positive_direction
negative_direction
```

where:

```text
positive_direction label = 1
negative_direction label = 0
```

This is the exact line where the antisymmetric relation is created.

## 26. Antisymmetry Generator: `assign_antisymmetry_split()`

Function:

```python
def assign_antisymmetry_split(
    *,
    sampled_pairs: list[tuple[str, str, int, int]],
    validation_fraction: float,
    split_mode: str,
    demo_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
```

This function is the antisymmetry equivalent of `assign_symmetry_split()`.

It first selects demo indices.

Then it defines query indices.

Then it samples query pairs for validation:

```python
validation_indices = set(rng.sample(query_indices, int(round(len(query_indices) * validation_fraction))))
```

Inside the loop, it creates:

```python
positive_direction, negative_direction = directed_labels(left, right, left_rank, right_rank)
```

Then it randomly chooses which direction is shown in train:

```python
if rng.random() < 0.5:
    train_direction = positive_direction
    reverse_direction = negative_direction
else:
    train_direction = negative_direction
    reverse_direction = positive_direction
```

This is important.

It prevents a trivial shortcut where train rows are always positive and reverse rows are always negative.

The pair role is:

```python
pair_role = "demo" if zero_based_pair_idx in demo_indices else "query"
```

The key split line is:

```python
reverse_split = "train" if pair_role == "demo" else "validation" if zero_based_pair_idx in validation_indices else "test"
```

Read it as:

- demo pair: both directions are train rows;
- query validation pair: train direction goes to train, reverse direction goes to validation;
- query test pair: train direction goes to train, reverse direction goes to test.

Then the function appends two atom rows:

```text
train_direction, split=train
reverse_direction, split=reverse_split
```

For a demo pair:

```text
(A, B), 1, train
(B, A), 0, train
```

or the opposite direction first, depending on the random choice.

For a query pair:

```text
(A, B), 1, train
(B, A), 0, validation/test
```

or:

```text
(B, A), 0, train
(A, B), 1, validation/test
```

The direction shown in train is random.

That means the model must use the relationship between a pair and its reverse, not just assume all held-out reverses have one fixed label.

## 27. Antisymmetry Generator: `validate_dataset()`

This validator checks that every reverse row exists and has the opposite label.

The key check is:

```python
elif reverse_label == label:
    antisymmetry_violation_examples.append((entity_1, entity_2))
```

For antisymmetry, same reverse label is an error.

The dataset is valid only if:

- required columns exist;
- forbidden columns are absent;
- every directed row has its reverse;
- no reverse pair has the same label;
- train split is non-empty;
- test split is non-empty.

The validator also returns:

- split counts;
- label counts;
- split-label counts;
- examples of missing reverse rows;
- examples of antisymmetry violations.

## 28. Antisymmetry Generator: `main()`

The orchestration is:

```text
parse arguments
create seeded random generator
build authors with hidden seniority ranks
build all unordered author pairs
sample unordered pairs
assign train/validation/test split
validate dataset
assemble metadata
write CSV, JSON, and Markdown outputs
raise error if validation fails
print summary
```

The model-facing file is:

```text
antisymmetry_coauthor_atoms.csv
```

The audit files are:

```text
entities.csv
pair_assignments.csv
metadata.json
validation_report.md
```

The important scientific guarantee is:

```text
hidden_seniority_rank is used to generate labels, but it is not written to antisymmetry_coauthor_atoms.csv.
```

## 29. Runner: File Purpose

File:

```text
run_minimal_symmetry_tfm_experiment.py
```

Despite the filename, this runner can evaluate both symmetry and antisymmetry datasets because both use the same four-column schema.

The runner performs binary classification on ordered entity pairs.

It can run:

- `tabpfn`;
- `tabicl`;
- `distmult`.

It writes:

- `metrics.csv`;
- `predictions.csv`;
- `metadata.json`;
- `findings.md`.

## 30. Runner: `MODELS`

At the top:

```python
MODELS = ("tabpfn", "tabicl", "distmult")
```

This tuple defines the valid model names accepted by `--models`.

If you pass a model not in this tuple, `validate_args()` raises an error.

## 31. Runner: `parse_args()`

The runner's command-line interface includes four groups of arguments.

Dataset and output arguments:

- `--data-path`;
- `--output-dir`.

Model selection and device arguments:

- `--models`;
- `--device`;
- `--seed`.

Threshold arguments:

- `--decision-threshold`;
- `--threshold-mode`.

Model-specific arguments:

- TabPFN checkpoint and inference settings;
- TabICL checkpoint and inference settings;
- DistMult training hyperparameters.

The default data path points to the symmetry dataset:

```python
Path("minimal_symmetry_dataset") / "symmetry_friendship_atoms.csv"
```

For antisymmetry, you must explicitly pass:

```text
--data-path minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv
```

## 32. Runner: `parse_csv_list()`

Function:

```python
def parse_csv_list(text: str) -> list[str]:
```

This converts a comma-separated model list into Python strings.

Example:

```text
--models tabpfn,tabicl,distmult
```

becomes:

```python
["tabpfn", "tabicl", "distmult"]
```

It strips spaces and ignores empty pieces.

## 33. Runner: `parse_bool_or_auto()`

Function:

```python
def parse_bool_or_auto(text: str) -> bool | str:
```

This is used for TabICL options such as:

- `--tabicl-use-amp`;
- `--tabicl-use-fa3`.

It accepts:

```text
auto
true
false
1
0
yes
no
```

and returns either:

- the string `"auto"`;
- Python `True`;
- Python `False`.

This keeps the command line flexible while passing the right type to TabICL.

## 34. Runner: `validate_args()`

Function:

```python
def validate_args(args: argparse.Namespace) -> list[str]:
```

This function checks the run plan before importing heavy dependencies.

It validates:

- selected models are known;
- at least one model was selected;
- the dataset path exists;
- the threshold is in `[0, 1]`;
- required checkpoints exist for selected TFM models.

The checkpoint checks are skipped for `--dry-run`.

This is intentional. Dry-run should be usable on a machine without TabPFN or TabICL installed.

If you run a real TabPFN experiment without a checkpoint:

```text
--models tabpfn
```

then `--tabpfn-model-path` is required.

Same for TabICL.

DistMult does not need an external checkpoint because it trains from scratch.

This follows the repository rule that the code must not fake missing models or silently skip unavailable dependencies.

## 35. Runner: `import_runtime_dependencies()`

Function:

```python
def import_runtime_dependencies() -> None:
```

This imports dependencies that are needed for real execution:

- NumPy;
- pandas;
- scikit-learn metrics.

It assigns them to global names:

```python
global np
global pd
global accuracy_score
...
```

Why import this way?

Because the script supports `--dry-run`.

With this design, dry-run can validate the CSV and print a plan without importing NumPy, pandas, scikit-learn, TabPFN, TabICL, or PyTorch.

If a dependency is missing in a real run, the script exits with a clear message explaining what is missing.

## 36. Runner: `load_dataset_rows()`

Function:

```python
def load_dataset_rows(path: Path) -> list[dict[str, str]]:
```

This is the first place where the runner takes the data.

It opens the CSV:

```python
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
```

So the data enters the experiment from:

```text
--data-path
```

The function then validates the table.

Required columns:

```python
required = {"entity_1", "entity_2", "label", "split"}
```

Forbidden leakage columns:

```python
forbidden = sorted({"hidden_group_id", "source_group_id", "target_group_id", "group_id"} & actual_columns)
```

Allowed splits:

```text
train
validation
test
```

Allowed labels:

```text
0
1
```

It also checks that train and test are both non-empty and both contain both classes.

This is a minimum sanity check. It does not prove the split is scientifically strong. It only prevents obviously broken input.

## 37. Runner: Entity Vocabulary

In `main()`, after loading rows:

```python
entity_categories = sorted({row["entity_1"] for row in rows} | {row["entity_2"] for row in rows})
```

This creates one shared vocabulary of all entities appearing anywhere in the dataset.

The same vocabulary is used for:

- train;
- validation;
- test.

This matters because categorical encodings must be consistent.

If `entity_0007` has code `6` in train, it must also have code `6` in test.

## 38. Runner: Split Construction

The runner does not create the split.

It only reads the split already written by the generator:

```python
train_rows = [row for row in rows if row["split"] == "train"]
valid_rows = [row for row in rows if row["split"] == "validation"]
test_rows = [row for row in rows if row["split"] == "test"]
```

This is important.

The generator decides which reverse pairs become train, validation, or test.

The runner just follows the CSV.

`pair_role` is not used here. It is not even present in the model-facing table.

Demo rows become train rows only because the generator wrote:

```text
split=train
```

for those rows.

## 39. Runner: `--dry-run`

If `--dry-run` is passed, the runner prints:

- resolved data path;
- selected models;
- number of entities;
- number of train rows;
- number of validation rows;
- number of test rows;
- train label counts;
- test label counts.

Then it returns before importing runtime dependencies.

Dry-run does not fit models.

Dry-run is useful for checking that the dataset path is correct and the split has the expected size.

## 40. Runner: DataFrame Conversion

In a real run, after importing dependencies:

```python
df = pd.DataFrame(rows)
df["label"] = df["label"].astype(int)
```

Then it creates split-specific DataFrames:

```python
train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
valid_df = df.loc[df["split"] == "validation"].reset_index(drop=True)
test_df = df.loc[df["split"] == "test"].reset_index(drop=True)
```

`reset_index(drop=True)` removes the old row numbers, so each split starts at index `0`.

This is not scientifically important, but it avoids confusing indices when writing predictions.

## 41. Runner: `make_feature_frame()`

Function:

```python
def make_feature_frame(df: Any, entity_categories: list[str]) -> Any:
```

This creates the actual feature matrix used by TabICL and then encoded for TabPFN and DistMult.

It defines a shared categorical dtype:

```python
entity_dtype = pd.CategoricalDtype(categories=entity_categories, ordered=False)
```

Then it creates a DataFrame with only:

```text
entity_1
entity_2
```

The target `label` is not included in `X`.

The split column is not included in `X`.

Hidden audit columns cannot be included because they were already rejected by `load_dataset_rows()`.

The returned frame has two categorical columns.

This is the complete feature representation given to the models.

## 42. Runner: Target Vectors

The labels are extracted separately:

```python
y_train = train_df["label"].astype(int).to_numpy()
y_valid = valid_df["label"].astype(int).to_numpy()
y_test = test_df["label"].astype(int).to_numpy()
```

These arrays are binary vectors.

The model fit receives:

```text
X_train, y_train
```

The final evaluation compares:

```text
y_test vs y_pred
```

## 43. Runner: `encode_for_tabpfn()`

Function:

```python
def encode_for_tabpfn(X: Any) -> Any:
```

This converts pandas categorical entity columns into integer category codes.

It returns:

```python
np.column_stack(
    [
        X["entity_1"].cat.codes.to_numpy(dtype=np.int64),
        X["entity_2"].cat.codes.to_numpy(dtype=np.int64),
    ]
)
```

So a row like:

```text
entity_0042, entity_0007
```

might become:

```text
41, 6
```

The output shape is:

```text
n_rows x 2
```

Both columns remain categorical conceptually, but TabPFN receives them as integer-coded categories.

## 44. Runner: `build_tabpfn_classifier()`

Function:

```python
def build_tabpfn_classifier(args: argparse.Namespace):
```

This constructs a `TabPFNClassifier`.

Important settings:

```python
model_path=str(args.tabpfn_model_path)
device=args.device
n_estimators=args.tabpfn_n_estimators
categorical_features_indices=[0, 1]
ignore_pretraining_limits=True
fit_mode=args.tabpfn_fit_mode
```

The key line is:

```python
categorical_features_indices=[0, 1]
```

This tells TabPFN that both columns are categorical features.

Without this, integer entity codes might be interpreted as ordered numeric values, which would be wrong.

## 45. Runner: `build_tabicl_classifier()`

Function:

```python
def build_tabicl_classifier(args: argparse.Namespace):
```

This constructs a `TabICLClassifier`.

Important settings:

```python
model_path=str(args.tabicl_model_path)
allow_auto_download=False
device=args.device
n_estimators=args.tabicl_n_estimators
batch_size=args.tabicl_batch_size
```

`allow_auto_download=False` is deliberate.

It means the script will not silently download a model checkpoint. The checkpoint must exist locally.

This makes runs reproducible and consistent with the rule that missing dependencies should not be faked.

## 46. Runner: Probability Extraction

The runner uses:

```python
extract_positive_proba(proba, classes)
```

instead of assuming the positive class is always in column `1`.

The function finds where `classes == 1`:

```python
positive_matches = np.where(classes == 1)[0]
```

Then it returns the probability column for class `1`.

This avoids a common bug.

Some classifiers may store classes as:

```text
[0, 1]
```

but robust code should not rely on that without checking.

## 47. Runner: `best_f1_threshold()`

Function:

```python
def best_f1_threshold(y_true: np.ndarray, positive_proba: np.ndarray, default_threshold: float) -> float:
```

This function is used only if:

```text
--threshold-mode tune_validation_f1
```

It searches candidate thresholds and chooses the one with best validation F1.

Candidate thresholds include:

- 0.00, 0.01, ..., 1.00;
- all predicted validation probabilities;
- the default threshold.

If validation has no rows or only one class, it returns the default threshold.

Important: threshold tuning uses validation data only. Final metrics are still computed on test data.

## 48. Runner: `evaluate_predictions()`

Function:

```python
def evaluate_predictions(y_true, y_pred, positive_proba) -> dict[str, float]:
```

This computes:

- accuracy;
- F1;
- precision;
- recall;
- true positive rate;
- predicted positive rate;
- average precision;
- ROC AUC.

Accuracy, F1, precision, and recall use binary predictions.

Average precision and ROC AUC use probabilities.

This distinction matters.

A model can rank positives above negatives reasonably well but still perform badly at threshold `0.5`.

That would show up as:

```text
decent average_precision or roc_auc
poor fixed-threshold F1
```

## 49. Runner: DistMult Motivation

DistMult is included as a relational baseline.

It learns:

- an embedding vector for each entity;
- one relation vector;
- one scalar bias.

The score is:

```text
score(A, B) = sum(embedding(A) * relation * embedding(B)) + bias
```

This score is symmetric because multiplication is commutative:

```text
embedding(A) * relation * embedding(B)
```

has the same elementwise product as:

```text
embedding(B) * relation * embedding(A)
```

Therefore:

```text
score(A, B) = score(B, A)
```

This makes DistMult a strong positive control for the symmetry dataset.

It also makes DistMult a deliberately wrong model for the antisymmetry dataset.

For antisymmetry, the desired behavior is:

```text
score(A, B) high
score(B, A) low
```

DistMult cannot naturally do that for a single relation because its score is symmetric.

## 50. Runner: `fit_predict_distmult()`

Function:

```python
def fit_predict_distmult(...):
```

This function trains DistMult from scratch.

It imports PyTorch inside the function:

```python
import torch
```

This means PyTorch is required only if `distmult` is selected.

The function validates DistMult hyperparameters:

- embedding dimension must be positive;
- epoch count must be positive;
- learning rate must be positive;
- weight decay must be non-negative;
- batch size must be positive;
- patience must be non-negative.

Then it checks CUDA:

```python
if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(...)
```

So if you request `--device cuda` on a machine without CUDA, the script fails explicitly.

## 51. Runner: DistMult Encoding

DistMult uses the same integer entity codes as TabPFN.

The code is:

```python
X_train_np = encode_for_tabpfn(X_train)
X_valid_np = encode_for_tabpfn(X_valid) if len(X_valid) else np.empty((0, 2), dtype=np.int64)
X_test_np = encode_for_tabpfn(X_test)
```

This produces arrays of shape:

```text
n_rows x 2
```

Each row is:

```text
left_entity_code, right_entity_code
```

Then it estimates the number of entities:

```python
n_entities = int(max(X_train_np.max(), X_test_np.max(), X_valid_np.max() if len(X_valid_np) else 0) + 1)
```

This is used to size the embedding table.

## 52. Runner: DistMult Model Class

Inside `fit_predict_distmult()`, the class is defined:

```python
class DistMultBinaryClassifier(torch.nn.Module):
```

It has three trainable components.

Entity embedding table:

```python
self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
```

Relation vector:

```python
self.relation = torch.nn.Parameter(torch.empty(embedding_dim))
```

Bias:

```python
self.bias = torch.nn.Parameter(torch.zeros(()))
```

Initialization:

```python
torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
torch.nn.init.normal_(self.relation, mean=0.0, std=1.0 / embedding_dim**0.5)
```

The `forward()` method receives a tensor of pairs:

```text
batch_size x 2
```

It looks up embeddings:

```python
left = self.entity_embeddings(pairs[:, 0])
right = self.entity_embeddings(pairs[:, 1])
```

Then scores the pair:

```python
return (left * self.relation * right).sum(dim=1) + self.bias
```

This returns raw logits, not probabilities.

## 53. Runner: DistMult Loss And Optimizer

The optimizer is:

```python
torch.optim.AdamW
```

The loss is:

```python
torch.nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss` expects raw logits.

So the model should not apply `sigmoid()` during training.

The sigmoid is applied later only for probabilities:

```python
torch.sigmoid(test_logits)
```

## 54. Runner: DistMult Tensors

The training arrays become tensors:

```python
train_pairs = torch.as_tensor(X_train_np, dtype=torch.long, device=device)
train_labels = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
```

Why `long` for pairs?

Because embedding lookup indices in PyTorch must be integer index tensors.

Why `float32` for labels?

Because binary cross-entropy expects floating labels:

```text
0.0 or 1.0
```

not integer class IDs.

Validation and test pairs are also converted to tensors.

## 55. Runner: DistMult Training Loop

The loop runs up to:

```python
args.distmult_epochs
```

For each epoch:

```python
model.train()
permutation = torch.randperm(len(train_pairs), device=device)
```

The permutation shuffles training rows.

Then mini-batches are processed:

```python
for start in range(0, len(train_pairs), batch_size):
    batch_idx = permutation[start : start + batch_size]
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_pairs[batch_idx])
    loss = loss_fn(logits, train_labels[batch_idx])
    loss.backward()
    optimizer.step()
```

This is a standard PyTorch supervised training loop:

- clear old gradients;
- compute logits;
- compute loss;
- backpropagate;
- update parameters.

After the epoch, the model is evaluated without gradients:

```python
model.eval()
with torch.no_grad():
```

The monitored loss is validation loss if validation rows exist:

```python
monitor_loss = loss_fn(model(valid_pairs), valid_labels)
```

Otherwise it falls back to train loss.

## 56. Runner: DistMult Early Stopping

The function keeps the best model state:

```python
best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
```

It updates this state when validation loss improves by at least `1e-7`:

```python
if monitor_loss < best_loss - 1e-7:
```

If loss does not improve:

```python
epochs_without_improvement += 1
```

and if patience is exceeded:

```python
break
```

At the end, the best state is loaded back into the model.

So predictions are made using the best validation-loss checkpoint, not necessarily the final epoch.

## 57. Runner: DistMult Prediction

After fitting:

```python
model.eval()
with torch.no_grad():
    valid_logits = model(valid_pairs)
    test_logits = model(test_pairs)
```

The logits become probabilities:

```python
valid_proba = torch.sigmoid(valid_logits).detach().cpu().numpy().astype(np.float64)
test_proba = torch.sigmoid(test_logits).detach().cpu().numpy().astype(np.float64)
```

Then thresholding happens:

```python
y_pred = (test_proba >= threshold).astype(np.int64)
```

The function returns probabilities, predictions, threshold, fit time, and predict time.

## 58. Runner: `fit_predict_one_model()`

This function dispatches to the selected model.

If the model is `distmult`, it immediately calls:

```python
fit_predict_distmult(...)
```

If the model is `tabpfn`, it:

- builds a TabPFN classifier;
- encodes categorical features as integer codes;
- passes integer arrays to TabPFN.

If the model is `tabicl`, it:

- builds a TabICL classifier;
- passes pandas categorical frames directly.

Then for TabPFN and TabICL it runs:

```python
model.fit(X_train_in, y_train)
```

and:

```python
model.predict_proba(X_test_in)
```

This is where the TFMs take data and make predictions.

They are fit on the rows where:

```text
split=train
```

They predict on the rows where:

```text
split=test
```

Validation rows are used only for threshold tuning if requested.

## 59. Runner: `build_findings()`

This function creates a compact Markdown summary from `metrics.csv`.

It writes a table with:

- model;
- accuracy;
- F1;
- average precision;
- ROC AUC;
- threshold.

Note: the title says `Minimal Symmetry TFM Findings` even when the input dataset is antisymmetry. That title is not methodologically important, but it is a naming imperfection in the current script.

## 60. Runner: `main()`

The runner orchestration is:

```text
parse args
validate args
load rows from --data-path
build entity vocabulary
partition rows by split
if dry-run: print plan and exit
import runtime dependencies
create DataFrames
build categorical feature frames
extract labels
create output directory
for each selected model:
    fit model
    predict test probabilities
    threshold probabilities
    compute metrics
    collect predictions
write metrics.csv
write predictions.csv
write metadata.json
write findings.md
print metrics table
```

This is the complete experiment pipeline.

## 61. What Exactly Becomes Train Data

The train data is exactly:

```python
train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
```

Then:

```python
X_train = make_feature_frame(train_df, entity_categories)
y_train = train_df["label"].astype(int).to_numpy()
```

For TabPFN:

```python
model.fit(encode_for_tabpfn(X_train), y_train)
```

For TabICL:

```python
model.fit(X_train, y_train)
```

For DistMult:

```python
train_pairs = torch.as_tensor(encode_for_tabpfn(X_train), ...)
train_labels = torch.as_tensor(y_train, ...)
```

So the generator controls ICL/training evidence by deciding which rows have `split=train`.

## 62. What Exactly Becomes Test Data

The test data is exactly:

```python
test_df = df.loc[df["split"] == "test"].reset_index(drop=True)
```

Then:

```python
X_test = make_feature_frame(test_df, entity_categories)
y_test = test_df["label"].astype(int).to_numpy()
```

For TabPFN:

```python
test_proba = predict_positive_proba(model, encode_for_tabpfn(X_test))
```

For TabICL:

```python
test_proba = predict_positive_proba(model, X_test)
```

For DistMult:

```python
test_logits = model(test_pairs)
test_proba = torch.sigmoid(test_logits)
```

The final metrics are computed only on `y_test`.

## 63. How Demo Split Becomes In-Context Evidence

The runner does not know the concept of `demo`.

The only mechanism is the `split` column.

In the symmetry generator:

```python
holdout_split = "train" if pair_role == "demo" else ...
```

In the antisymmetry generator:

```python
reverse_split = "train" if pair_role == "demo" else ...
```

So demo pairs are converted into train rows during generation.

By the time the runner reads the CSV, demo rows are just ordinary train rows.

This means:

```text
demo fraction controls how many explicit reverse examples are placed into the model's fit context.
```

## 64. How To Read `pair_assignments.csv`

`pair_assignments.csv` is an audit file.

It is not passed to the model.

For symmetry, it tells you:

- which unordered pair was sampled;
- whether it was `demo` or `query`;
- which direction was train;
- which direction was reverse;
- whether the reverse was train, validation, or test.

For antisymmetry, it additionally tells you:

- which direction is positive;
- which direction is negative;
- which direction was shown in train;
- which reverse direction was held out.

Use this file to verify the split design.

Do not use it as a model input.

## 65. Scientific Interpretation Of The Two Split Modes

`reverse_holdout` is the stricter mode.

For every pair, the model sees one direction and must predict the reverse direction.

There are no explicit train examples showing the rule on other pairs.

This asks:

```text
Can the model infer the reverse-pair rule from the raw representation and labels alone?
```

`symmetry_demonstration` and `antisymmetry_demonstration` are ICL-friendly modes.

Some pairs explicitly show both directions in train.

This asks:

```text
Can the model use explicit demonstrations of the relation property and apply them to query pairs?
```

For tabular foundation models, the demonstration mode is often the fairer ICL test.

But it is still not a full symbolic generalization test. The entities in train and test belong to the same generated world.

## 66. Why DistMult Results Are Useful

For the symmetry dataset, DistMult should perform well because its scoring function is symmetric.

If DistMult gets high performance while TFMs fail, that tells us:

```text
The dataset is solvable, but the TFMs are not exploiting the needed pair-reversal structure.
```

For the antisymmetry dataset, DistMult should perform badly because it cannot naturally assign opposite scores to reverse pairs.

If DistMult fails while TFMs perform above chance, that tells us:

```text
The antisymmetry dataset is not being solved by a symmetric scoring shortcut.
```

DistMult is not a universally fair baseline. It is an inductive-bias diagnostic.

## 67. Validity Checklist

Before trusting a run, check these files.

Model-facing CSV:

- contains only `entity_1`, `entity_2`, `label`, `split`;
- has train and test rows;
- has both labels in train and test;
- has no hidden group or hidden rank columns.

Generator metadata:

- `entity_ids_randomized` should usually be `true`;
- `split_mode` should match the intended protocol;
- `pair_role_counts` should show nonzero query pairs;
- validation status should be true.

Pair assignments:

- demo pairs should have both directions in train;
- query pairs should have one direction in train and reverse in validation/test;
- symmetry reverse labels should match;
- antisymmetry reverse labels should differ.

Runner outputs:

- `metrics.csv` should contain one row per selected model;
- `predictions.csv` should contain test rows only, repeated once per model;
- `metadata.json` should point to the intended input dataset;
- `findings.md` should agree with `metrics.csv`.

## 68. Important Limitations

The current datasets test ordered-pair generalization inside one synthetic entity universe.

They do not prove that a model has learned an abstract symbolic law that transfers to a completely new universe.

The train and test rows share the same entity vocabulary.

This means the experiment is mainly about:

```text
Can a model use observed pair evidence to predict held-out reverse directions?
```

It is not yet:

```text
Can a model learn symmetry or antisymmetry as an abstract rule and transfer it to unseen relation matrices?
```

That stronger claim would require a matrix-out or universe-out protocol where entire relation worlds or entity sets are held out.

## 69. Minimal Commands

Generate the symmetry dataset:

```bash
python generate_minimal_symmetry_dataset.py \
  --output-dir minimal_symmetry_dataset \
  --split-mode symmetry_demonstration \
  --demo-fraction 0.5
```

Generate the antisymmetry dataset:

```bash
python generate_minimal_antisymmetry_dataset.py \
  --output-dir minimal_antisymmetry_dataset \
  --split-mode antisymmetry_demonstration \
  --demo-fraction 0.5
```

Dry-run the runner:

```bash
python run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_symmetry_dataset/symmetry_friendship_atoms.csv \
  --models distmult \
  --dry-run
```

Run DistMult on symmetry:

```bash
python run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_symmetry_dataset/symmetry_friendship_atoms.csv \
  --output-dir results_symmetry_distmult \
  --models distmult \
  --device cuda
```

Run TFMs on antisymmetry:

```bash
python run_minimal_symmetry_tfm_experiment.py \
  --data-path minimal_antisymmetry_dataset/antisymmetry_coauthor_atoms.csv \
  --output-dir results_antisymmetry_tfms \
  --models tabpfn,tabicl \
  --device cuda \
  --tabpfn-model-path /path/to/tabpfn/checkpoint \
  --tabicl-model-path /path/to/tabicl/checkpoint
```

## 70. Mental Model

Think of the generators as the experimental design.

They answer:

```text
What hidden world exists?
What labels are true?
Which evidence rows are shown to the model?
Which reverse rows are held out for evaluation?
```

Think of the runner as the measurement instrument.

It answers:

```text
Given this model-facing CSV, how well does each selected model predict the test labels?
```

If a result looks surprising, audit the generator first, then audit the runner.

The most important question is always:

```text
What information was actually available in entity_1 and entity_2 at fit time?
```
