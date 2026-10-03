# Classical machine learning

Three end-to-end tabular pipelines: feature engineering, preprocessing, model
selection and evaluation.

---

## 1. Classify song genres from audio data

**File:** [`01-classify-song-genres-pca.ipynb`](01-classify-song-genres-pca.ipynb)
**Status:** ✅ runs end to end
**Competencies:** dimensionality reduction ●●● · model comparison ●● · class imbalance ●· cross-validation ●·

Rock vs hip-hop, from 52 pre-extracted audio features (the echonest set) joined
to track metadata.

**Approach:** inspect the correlation matrix to find redundant features →
standard-scale → PCA, choosing components by cumulative explained variance
(90% threshold) → train a decision tree and a logistic regression →
rebalance the classes → re-evaluate with 10-fold K-fold cross-validation.

### ⚠️ The reported comparison is not what the code computes

Step 10 reads:

```python
logit_score = cross_val_score(tree, pca_projection, labels, cv=kf)
```

`logit_score` is a *second* cross-validation of the **decision tree**. The
logistic regression is never cross-validated, so the printed "Decision Tree vs
Logistic Regression" comparison compares the decision tree to itself. The fix
is one word — `logreg` instead of `tree` — but until then, treat the
conclusion as unverified.

The method is sound; the reporting is not. Everything before step 10 is valid.

### Reproducing

```bash
pip install numpy pandas scikit-learn matplotlib
python ../../scripts/fetch_data.py --only song-genres
jupyter notebook 01-classify-song-genres-pca.ipynb
```

---

## 2. Predicting credit card approvals

**File:** [`02-credit-card-approvals.ipynb`](02-credit-card-approvals.ipynb)
**Status:** ✅ runs
**Competencies:** data wrangling ●●● · preprocessing ●●● · hyperparameter search ●·

Binary classification of credit-card applications, from the UCI Credit
Approval dataset.

**The strongest data-wrangling work in the repository.** The notebook inspects
missingness *before* choosing a strategy, then splits the imputation by column
type: **mean** for numeric columns, **mode** for categorical ones. That is a
defensible decision made from the data's shape, rather than one blanket fill.

Also demonstrates: catching the `?` sentinel before it reaches the model,
label encoding, min-max scaling, dropping low-value features, and `GridSearchCV`
over `tol` and `max_iter`.

### The leakage bug, and what honest reporting revealed

The original cell computed the scaling and then never used it:

```python
rescaledX = scaler.fit_transform(X)        # fitted on ALL of X, test included
grid_model_result = grid_model.fit(X, y)   # ...and passed raw X anyway
```

So the scaler leaked test statistics, and the scaled array was discarded while
the search ran on unscaled data. It is now a `Pipeline`, which fixes both and
makes the preprocessing travel with the model.

Reporting the numbers honestly rather than quoting `best_score` alone is what
makes the value of the fix visible:

```
Best: 0.852174 using {'logreg__max_iter': 100, 'logreg__tol': 0.01}
score across 9 configurations: min=0.8507 max=0.8522 mean=0.8512
fold-to-fold std at the best config: 0.1372
tuned pipeline on the held-out test split: 0.8296 (+/- 0.0613)
```

Cross-validation claims 0.852. The model actually scores **0.830** on held-out
data, with a fold-to-fold standard deviation of 0.137 — so most of the apparent
tuning gain was noise. The original bug concealed that.

### Reproducing

### Reproducing

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook 02-credit-card-approvals.ipynb
```

Dataset is committed at
[`data/machine-learning/cc_approvals.data`](../../data/machine-learning/cc_approvals.data).

---

## 3. Automatic model selection with TPOT

**File:** [`03-tpot-blood-donations-automl.ipynb`](03-tpot-blood-donations-automl.ipynb)
**Status:** ✅ runs
**Competencies:** AutoML ●●● · evaluation metrics ●●● · feature scaling ●·

Predicting whether a blood donor will donate again, from RFM (recency,
frequency, monetary) features. The target has roughly 5% incidence, which
drives most of the decisions here.

**Approach:** check target incidence → stratified train/test split → hand the
search to **TPOT**, which uses genetic programming to evolve whole pipelines →
score with **AUC** rather than accuracy, which would be near-useless at 5%
positive rate → inspect variance, find `Monetary (c.c. blood)` dominating it,
apply a log transform → compare against a hand-built logistic regression.

### What it demonstrates

- Choosing a metric appropriate to an imbalanced target. This is the single
  most consequential decision in the notebook, and it was made correctly.
- Using stratified splitting to keep the positive rate stable across splits.
- Justifying a transform from observed variance rather than by habit.
- Automated ML as a tool: TPOT returned logistic regression with no
  preprocessing, which the log transform then beat.

The cell was rewritten for the **TPOT 1.x** API — `generations`,
`population_size`, `verbosity` and `config_dict` no longer exist, and passing
them raises `TypeError`. The search is now expressed as `search_space`,
`scorers`, `max_time_mins` and `early_stop`.

**Caveat:** the AUC is still reported as a point estimate with no confidence
interval. A single number without uncertainty is weak evidence for a small
improvement.

### Reproducing

```bash
pip install numpy pandas scikit-learn tpot
jupyter notebook 03-tpot-blood-donations-automl.ipynb
```

TPOT searches take a while. Expect minutes, not seconds.

Dataset is committed at
[`data/machine-learning/transfusion.data`](../../data/machine-learning/transfusion.data).
