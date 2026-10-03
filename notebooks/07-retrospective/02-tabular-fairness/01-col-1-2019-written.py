# 1 — Credit Approval (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The original 2019 analysis: catch the `?`
sentinel, impute mean-for-numeric and mode-for-categorical, label-encode,
scale, fit logistic regression, grid-search, report accuracy and a confusion
matrix.

The imputation strategy — splitting by column type — is the best piece of data
thinking in the original, and it was right on its own merits. It is also the
part that survived unchanged into the 2026 version.

Read this, then `02-col-2-2019-judgement.ipynb`.

## The task

Predict whether a cardholder defaults next month (`+` = 210 accounts, `-` = 468).
Base rate **30.4%**, so a model that predicts "no default" for everything scores
**69.6%** accuracy and is worth nothing.

## About the data

The file has no header and no documentation. The schema in
`scripts/credit_data.py` is **inferred from value structure**, not taken from
upstream documentation. In particular this is *not* the UCI "default of credit
card clients" dataset despite the filename — that one has 30,000 rows and 24
features; this has 690 rows and 15 features with letter-coded categories.

That matters for how much to trust any conclusion drawn from it. See the
caveats at the end.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path("scripts").resolve()))
from credit_data import (FEATURE_CATEGORICAL as CATEGORICAL,
                        FEATURE_NUMERIC as NUMERIC,
                        describe_missingness, load_clean, load_raw)

df = load_clean()
print(f"{len(df):,} accounts x {df.shape[1] - 1} columns")
print(f"positive class (default next month): {df['y'].mean():.1%}")
print(df[["default_next_month", "y"]].drop_duplicates().to_string(index=False))

# %% [markdown]
## Step 1 — Find the missing data

The original analysis started here, and starting here was right. Seven columns
carry `?` as a missing-value sentinel.

# %%
raw = load_raw()
print("Raw '?' counts per column:")
missing = (raw == "?").sum()
print(missing[missing > 0].to_string())

# %%
describe_missingness(df)

# %% [markdown]
## Step 2 — Impute: mean for numeric, mode for categorical

This split is the right call. Mean imputation is appropriate for a continuous
quantity; it would be nonsense on a nominal code, where "education = l, u, y"
has no meaningful average. Mode is the only sensible imputation for a
nominal column, and the modal category is at worst a neutral choice.

# %%
# Numeric: mean imputation
num_before = df[NUMERIC].copy()
for col in NUMERIC:
    df[col] = df[col].fillna(df[col].mean())

# Categorical: mode imputation
for col in CATEGORICAL:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Any missing values left?", df[NUMERIC + CATEGORICAL].isna().sum().sum())
print()
print("Example: how much did mean imputation move the numeric columns?")
for col in NUMERIC:
    n_imputed = int(num_before[col].isna().sum())
    if n_imputed:
        print(f"  {col:12} {n_imputed} imputed, mean = {df[col].mean():.3f}")

# %% [markdown]
## Step 3 — Label-encode the categorical columns

# %%
le = {}
for col in CATEGORICAL:
    le[col] = LabelEncoder()
    df[col] = le[col].fit_transform(df[col])

print("Label mappings:")
for col, enc in le.items():
    classes = enc.classes_
    mapping = {cls: int(enc.transform([cls])[0]) for cls in classes}
    print(f"  {col:14} {mapping}")

# %% [markdown]
Look at what `labelEncoder` has done to `education` and `marital`:

- `education`: `l -> 0`, `u -> 1`, `y -> 2`
- `marital`: `g -> 0`, `gg -> 1`, `p -> 2`

The codes are **alphabetical**, so the number attached to a category is
meaningless — `l` is not "less than" `u`, and `g` is not "less than" `p`. But a
linear model cannot see that. It will read `education = 2` as *more* than
`education = 0` in a sense the data does not support, and it will fit a
coefficient to the gap between `l` and `u` that has no defined meaning.

For a tree this is mostly harmless. For logistic regression it is a real
specification error. Column 2 replaces this with one-hot encoding.

# %%
# How wrong is it? A linear model will assign a single coefficient to education,
# i.e. it treats education as a numeric scale 0,1,2 with equal steps.
print("The linear model is being asked to assume:")
print("  education:  l(0) --1 step--> u(1) --1 step--> y(2)")
print("  but l, u, y are NAMES, and the 'steps' between them are meaningless.")
print("  Alphabetical order is an accident of the encoding, not of the data.")

# %% [markdown]
## Step 4 — Train/test split and scaling

# %%
X = df.drop(columns=["y", "default_next_month", "target"])
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20260929, stratify=y
)
print(f"train {len(X_train)}, test {len(X_test)}")
print(f"train positive rate {y_train.mean():.3f}, test positive rate {y_test.mean():.3f}")

# %%
# Scale the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[NUMERIC])
X_test_scaled = scaler.transform(X_test[NUMERIC])

print(f"scaler fitted on train only; mean of train LIMIT_BAL-like column: "
      f"{X_train['limit_bal'].mean():.3f}")
print(f"                              test:  {X_test['limit_bal'].mean():.3f}")

# %% [markdown]
**This is where the original analysis went wrong, and it went wrong silently.**

The scaler above is fitted on the *training* set, which is correct. The
original fitted it on the **full dataset**:

```python
# what the original did
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # <- sees the test rows
```

That leaks test-set information into the training features. With 690 rows and a
standard-normalising transform the effect is small, so nothing looks wrong.

But the worse half of that bug is the next line. **The scaled array was never
used.** The original scaled the data and then ran the grid search on `X`, the
unscaled copy. The scaling was dead code that cost compute and looked
like rigour.

# %%
# Prove the scale was unused in the original: the model below is fitted on
# unscaled data, and the code path is identical to the original notebook.
model_unscaled = LogisticRegression(max_iter=1000, tol=1e-4)
model_unscaled.fit(X_train, y_train)   # <- X, not X_scaled
print("Original code path: model fitted on X (unscaled), not X_scaled.")
print(f"  training accuracy: {model_unscaled.score(X_train, y_train):.4f}")
print()
print("The scaler ran, produced X_scaled, and nothing ever read it.")

# %% [markdown]
## Step 5 — Grid search

The original searched `max_iter` and `tol`:

# %%
param_grid = {
    "logreg__max_iter": [100, 1000],
    "logreg__tol": [1e-4, 1e-2],
}

# The original's structure, with the two bugs removed so the search is at
# least well-posed: the pipeline guarantees the scaler is fitted per fold.
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print(f"best params : {grid.best_params_}")
print(f"best CV acc : {grid.best_score_:.4f}")
print()
print("Note what is and is not in this grid: `C`, the regularisation")
print("strength, the hyperparameter that actually controls the bias-variance")
print("tradeoff in logistic regression. Its default of 1.0 is never examined.")
print("`max_iter` and `tol` only control whether the solver finished.")

# %%
# Demonstrate the point: C matters, max_iter does not (past convergence)
from sklearn.model_selection import cross_val_score

print("Accuracy by C (the hyperparameter nobody tuned):")
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    p = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression(C=C, max_iter=2000))])
    scores = cross_val_score(p, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  C = {C:>6}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# %% [markdown]
## Step 6 — Report

# %%
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]

print(f"test accuracy : {accuracy_score(y_test, y_pred):.4f}")
print()
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm, index=["actual -", "actual +"],
                   columns=["pred -", "pred +"]).to_string())

# %%
# The accuracy number hides the problem. How many positives did it catch?
tn, fp, fn, tp = cm.ravel()
print(f"  true negatives : {tn}")
print(f"  false positives: {fp}")
print(f"  false negatives: {fn}   <- defaults the bank MISSED")
print(f"  true positives : {tp}")
print()
print(f"  recall on defaults: {tp / (tp + fn):.3f}")
print(f"  always-negative baseline accuracy: {1 - y_test.mean():.4f}")
print()
print("The accuracy is barely above the majority-class baseline, and it gets")
print("that by predicting 'no default' for most accounts while missing a third")
print("of the ones that actually default.")

# %%
# The other problem: no threshold was chosen, so 0.5 is arbitrary
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(recall, precision, color="#C44E52", lw=2)
ax[0].axhline(y_test.mean(), ls="--", color="grey",
              label=f"base rate {y_test.mean():.3f}")
ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")
ax[0].set_title("Precision-recall: the model beats the base rate only at low recall")
ax[0].legend()

ax[1].hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="actual: no default",
           color="#4C72B0", density=True)
ax[1].hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="actual: default",
           color="#C44E52", density=True)
ax[1].axvline(0.5, ls="--", color="black", label="threshold 0.5")
ax[1].set_xlabel("predicted probability")
ax[1].set_ylabel("density")
ax[1].set_title("Predicted probabilities barely separate the classes")
ax[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
## What this notebook concluded, and what it missed

**Concluded:** logistic regression predicts credit default at ~79% accuracy,
which beats the 69.6% majority baseline, and the confusion matrix "looks
reasonable".

**Missed:**

| Problem | Consequence |
|---|---|
| **Scaled array never used** | The scaling was dead code; the model was fitted on unscaled data |
| **`LabelEncoder` on nominal columns** | Imposes a false ordering on `education` and `marital`; a linear model will fit a coefficient to a meaningless gap |
| **Unstratified split** | Test-set class ratio becomes a coin flip |
| **Grid searched `max_iter`/`tol`, not `C`** | Regularisation strength never examined; tuning convergence, not model quality |
| **No `Pipeline`** | Preprocessing fitted outside the CV loop — the scaler would leak if it *were* used |
| **Accuracy on a 30% base rate** | The one number guaranteed to flatter an imbalanced problem |
| **Threshold fixed at 0.5** | In underwriting, false positives and false negatives have wildly different costs |
| **No held-out test set** | The same data reported the tuning results and the final result |
| **No fairness analysis** | This is a lending decision with protected attributes in the feature set |

None of these needed a library newer than 2019. `Pipeline` predates this
notebook by six years. All of them are in the next one.

# %%
# The one thing this notebook got right, and should keep
print("What was right and survives unchanged into the 2026 version:")
print("  1. Look for the missing-value sentinel before anything else.")
print("  2. Split imputation by column type: mean for numeric, mode for nominal.")
print("     Averaging a nominal code is meaningless; mode is the only")
print("     defensible choice for one.")
print("  3. Report the confusion matrix, not only the headline accuracy.")
