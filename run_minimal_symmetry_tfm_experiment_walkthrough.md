# Walkthrough: `run_minimal_symmetry_tfm_experiment.py`

This document explains what `run_minimal_symmetry_tfm_experiment.py` does step by step. The goal is to make the fitting and prediction pipeline auditable.

The same script is used for both generated datasets because both expose the same model-facing schema:

```text
entity_1,entity_2,label,split
```

The script does not know whether the task is symmetry or antisymmetry. The dataset generator defines the relational property through the labels and the split.

## 1. Command-Line Inputs

The script starts by parsing arguments in `parse_args()`.

Important arguments:

- `--data-path`: CSV file to evaluate.
- `--output-dir`: folder where `metrics.csv`, `predictions.csv`, `metadata.json`, and `findings.md` are written.
- `--models`: comma-separated model list. Current valid values are `tabpfn`, `tabicl`, and `distmult`.
- `--device`: `cpu`, `cuda`, or another PyTorch-compatible device string for DistMult and the TFM wrappers.
- `--threshold-mode`: either `fixed` or `tune_validation_f1`.
- `--decision-threshold`: fixed classification threshold, default `0.5`.

For TabPFN and TabICL, the script requires explicit checkpoint paths:

```text
--tabpfn-model-path
--tabicl-model-path
```

DistMult does not need an external checkpoint. It is trained from scratch on the train split.

## 2. Argument Validation

`validate_args(args)` checks:

- requested models are valid;
- `--data-path` exists;
- decision threshold is in `[0, 1]`;
- TabPFN checkpoint exists if `tabpfn` is selected;
- TabICL checkpoint exists if `tabicl` is selected.

This follows the `AGENTS.md` rule: the script does not fake missing checkpoints or silently skip unavailable models.

If only `distmult` is selected, no TabPFN or TabICL checkpoint is required.

## 3. Dependency Loading

`import_runtime_dependencies()` imports runtime dependencies only after dry-run handling.

It imports:

- `numpy`;
- `pandas`;
- scikit-learn metrics.

DistMult imports `torch` inside `fit_predict_distmult()`.

This design lets `--dry-run` validate the CSV without requiring the full model environment.

## 4. Dataset Loading

The CSV is read in `load_dataset_rows(path)`.

The required columns are:

```text
entity_1
entity_2
label
split
```

The script rejects missing required columns.

It also rejects obvious leakage columns:

```text
hidden_group_id
source_group_id
target_group_id
group_id
```

Important: for the current generated datasets, hidden labels such as group membership or seniority rank are kept only in audit files like `entities.csv`, not in the model-facing CSV.

The script verifies:

- `split` contains only `train`, `validation`, and `test`;
- `label` is binary `0/1`;
- train split is non-empty;
- test split is non-empty;
- train split has both classes;
- test split has both classes.

## 5. Entity Vocabulary

After loading rows, the script builds one shared entity vocabulary:

```python
entity_categories = sorted({row["entity_1"] for row in rows} | {row["entity_2"] for row in rows})
```

This vocabulary is shared across train, validation, and test.

That matters because the same entity must receive the same categorical code in every split.

## 6. Split Construction

The script does not create the split. The generator already wrote the split column.

The evaluator simply partitions rows:

```python
train_rows = [row for row in rows if row["split"] == "train"]
valid_rows = [row for row in rows if row["split"] == "validation"]
test_rows = [row for row in rows if row["split"] == "test"]
```

Meaning:

- `train`: in-context/training rows passed to the model fit method;
- `validation`: optional threshold tuning rows;
- `test`: rows used for final prediction and metrics.

For demo splits, demo rows become train because the generator sets their `split` value to `train`. The evaluator does not see `pair_role`; it only sees `split`.

## 7. Feature Construction

After `--dry-run`, rows are converted to a pandas DataFrame.

Then:

```python
X_train = make_feature_frame(train_df, entity_categories)
X_valid = make_feature_frame(valid_df, entity_categories)
X_test = make_feature_frame(test_df, entity_categories)
```

`make_feature_frame()` keeps only two feature columns:

```text
entity_1
entity_2
```

Both columns are pandas categoricals with the same fixed category list.

The target vectors are:

```python
y_train = train_df["label"].astype(int).to_numpy()
y_valid = valid_df["label"].astype(int).to_numpy()
y_test = test_df["label"].astype(int).to_numpy()
```

No split column, hidden group column, hidden seniority column, or audit column is passed as a feature.

## 8. TabPFN Encoding

TabPFN receives integer categorical codes.

`encode_for_tabpfn(X)` converts:

```text
entity_1 categorical code
entity_2 categorical code
```

into a NumPy array with shape:

```text
n_rows x 2
```

Example:

```text
entity_0007 -> 6
entity_0042 -> 41
```

The TabPFN classifier is told that both columns are categorical:

```python
categorical_features_indices=[0, 1]
```

## 9. TabPFN Fit And Prediction

When `model_name == "tabpfn"`, the script:

1. builds a `TabPFNClassifier`;
2. encodes train, validation, and test features as integer categorical codes;
3. calls:

```python
model.fit(X_train_in, y_train)
```

4. gets probabilities with:

```python
model.predict_proba(X_test_in)
```

The positive class probability is extracted by `extract_positive_proba()`, which explicitly finds class `1` in `model.classes_`.

This avoids assuming class order.

## 10. TabICL Fit And Prediction

When `model_name == "tabicl"`, the script:

1. builds a `TabICLClassifier`;
2. passes pandas categorical feature frames directly;
3. calls:

```python
model.fit(X_train_in, y_train)
```

4. gets probabilities with:

```python
model.predict_proba(X_test_in)
```

As with TabPFN, the positive class probability is extracted by matching class `1`.

## 11. DistMult Baseline

When `model_name == "distmult"`, the script calls `fit_predict_distmult()`.

DistMult is an embedding model. Each entity gets a learned vector:

```text
embedding(entity)
```

There is also one learned relation vector:

```text
relation
```

The score for a pair is:

```text
score(entity_1, entity_2) =
sum(embedding(entity_1) * relation * embedding(entity_2)) + bias
```

This score is symmetric by construction because multiplication is commutative:

```text
score(A, B) = score(B, A)
```

This is why DistMult is a good positive-control model for the symmetry dataset.

It is also why DistMult is a deliberately bad model for antisymmetry: it cannot naturally assign opposite scores to `(A, B)` and `(B, A)`.

## 12. DistMult Training Details

DistMult uses PyTorch.

The script converts categorical entity codes to tensors:

```python
train_pairs = torch.as_tensor(X_train_np, dtype=torch.long, device=device)
train_labels = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
```

The loss is:

```python
torch.nn.BCEWithLogitsLoss()
```

This is binary cross-entropy applied to raw logits.

The optimizer is:

```python
torch.optim.AdamW
```

The training loop:

1. shuffles training rows every epoch;
2. trains in mini-batches;
3. evaluates validation loss after each epoch;
4. keeps the best validation-loss model state;
5. stops early after `--distmult-patience` epochs without improvement.

Relevant hyperparameters:

```text
--distmult-embedding-dim
--distmult-epochs
--distmult-lr
--distmult-weight-decay
--distmult-batch-size
--distmult-patience
```

Prediction uses:

```python
torch.sigmoid(test_logits)
```

to convert logits into probabilities.

## 13. Thresholding

All models return positive-class probabilities.

The script then converts probabilities into binary predictions.

Default mode:

```text
--threshold-mode fixed
```

uses:

```text
threshold = 0.5
```

Then:

```python
y_pred = (positive_proba >= threshold).astype(np.int64)
```

Optional mode:

```text
--threshold-mode tune_validation_f1
```

searches thresholds using the validation split and chooses the threshold with the best validation F1.

The final reported metrics are always computed on the test split.

## 14. Metrics

`evaluate_predictions()` computes:

- accuracy;
- F1;
- precision;
- recall;
- true positive rate in the test labels;
- predicted positive rate;
- average precision;
- ROC AUC.

Average precision and ROC AUC use the raw probabilities, not the thresholded predictions.

This matters because a model can have mediocre fixed-threshold accuracy but still rank positives above negatives well.

## 15. Output Files

The script writes four files under `--output-dir`.

### `metrics.csv`

One row per model:

```text
model,threshold,fit_time_s,predict_time_s,accuracy,f1,precision,recall,positive_rate_true,positive_rate_pred,average_precision,roc_auc
```

### `predictions.csv`

One row per test example per model:

```text
entity_1,entity_2,label,model,positive_proba,prediction
```

If optional ID columns like `atom_id` or `pair_id` exist in a dataset, they are preserved. The current minimal datasets do not include them in the model-facing CSV.

### `metadata.json`

Records:

- timestamp;
- data path;
- selected models;
- device;
- threshold mode;
- number of train, validation, and test rows.

### `findings.md`

A compact markdown table generated from `metrics.csv`.

## 16. What The Script Does Not Do

The script does not generate labels.

The script does not decide whether the task is symmetry or antisymmetry.

The script does not pass hidden group IDs or hidden seniority ranks to the models.

The script does not know `pair_role`.

The script only uses:

```text
entity_1
entity_2
label
split
```

Therefore, the validity of the relational test depends mostly on the generator and the split protocol.

## 17. Why DistMult Is A Valid Positive Control

For symmetry:

```text
label(A, B) = label(B, A)
```

DistMult has exactly this inductive bias. If DistMult performs well while TabPFN and TabICL fail, the dataset is not impossible. Rather, the raw ordered-pair representation is not enough for the TFMs to infer the symmetry rule.

For antisymmetry:

```text
label(A, B) = 1 - label(B, A)
```

DistMult has the wrong inductive bias. It should fail or perform poorly. If it does, that validates that the antisymmetry task is not being solved by a symmetric shortcut.

## 18. Current Result Interpretation

On the current symmetry dataset:

- DistMult reaches perfect performance.
- TabPFN and TabICL remain close to chance.

This supports the claim:

```text
The symmetry dataset is solvable, but the TFMs do not learn raw pair-reversal symmetry from this representation.
```

On the current antisymmetry dataset:

- TabPFN and TabICL are above chance.
- DistMult performs badly.

This supports the claim:

```text
The TFMs exploit some directional pairwise signal, while DistMult fails because symmetric scoring is the wrong bias for antisymmetry.
```

## 19. Main Audit Checklist

To validate a run, check:

- the input CSV contains only `entity_1`, `entity_2`, `label`, and `split`;
- hidden metadata appears only in audit files, not in the model-facing CSV;
- `metadata.json` says entity IDs were randomized;
- train/test both contain classes `0` and `1`;
- result files were produced under the intended `--output-dir`;
- `predictions.csv` contains only test rows;
- DistMult is used as a positive control for symmetry, not as a fair antisymmetry model.
