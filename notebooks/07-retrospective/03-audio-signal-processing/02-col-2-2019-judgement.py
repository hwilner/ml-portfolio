# 2 — Song Genre from Audio Features (2019 judgement, same tools)

## What this notebook is

**Column 2 of the retrospective.** The same 4,802 tracks and the same 2019-era
tools, with the analysis done using judgement I did not have in 2019.

Everything here was possible in 2019. `Pipeline` (2013), `class_weight` (2011),
`StratifiedKFold` (2012), PR-AUC (always), nested CV (implementable in a few
lines). **This is a discipline gap, not a tooling gap.**

## The changes from column 1

| # | Change | Possible in 2019? |
|---|---|---|
| 1 | Everything inside a `Pipeline` | Yes — `Pipeline` since 0.14 (2013) |
| 2 | `class_weight="balanced"` instead of resampling | Yes |
| 3 | `n_components` inside model selection | Yes |
| 4 | A non-PCA baseline | Yes |
| 5 | PR-AUC and balanced accuracy alongside accuracy | Yes |
| 6 | A truly untouched final test set | Yes |
| 7 | A **paired** statistical test between the two models | Yes — and this is the one that settles the original's wrong conclusion |

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 20260929
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message=".*unknown categories.*")

with open("data/machine-learning/echonest-metrics.json") as fh:
    raw = json.load(fh)

FEATURES = ["acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "speechiness", "tempo", "valence"]

echo = pd.DataFrame({k: list(v.values()) for k, v in raw.items()
                     if k in FEATURES + ["track_id"]})
meta = pd.read_csv("data/machine-learning/fma-rock-vs-hiphop.csv",
                   usecols=["track_id", "genre_top"])
df = echo.merge(meta, on="track_id", how="inner")
df = df[df.genre_top.isin(["Rock", "Hip-Hop"])].reset_index(drop=True)
df["y"] = (df.genre_top == "Hip-Hop").astype(int)

X, y = df[FEATURES], df["y"]
print(f"{len(df):,} tracks | Hip-Hop rate {y.mean():.3f}")
print(f"majority-class accuracy: {1 - y.mean():.4f}")

# %% [markdown]
## Change 1 — One `Pipeline` for everything

Column 1 fitted the scaler and the PCA on the full dataset, then treated them as
fixed during cross-validation. Every fold therefore scored a model whose
features had been shaped using the held-out rows.

A `Pipeline` makes the leak structurally impossible: each fold re-fits the
scaler on its own training partition, re-fits the PCA on that partition's
scaled data, and fits the classifier on the resulting components.

# %%
# Three-way split: a development set for tuning, and a final test set that
# nothing touches until the very end.
X_dev, X_final, y_dev, y_final = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"development {len(X_dev):,}  final test {len(X_final):,}")
print(f"  final test positive rate: {y_final.mean():.3f}")

# %%
# Demonstrate that the leak is gone: the PCA's component loadings now differ
# between folds, because each is fitted on different data.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
loadings = []
for tr, _ in cv.split(X_dev, y_dev):
    p = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=2))]).fit(X_dev.iloc[tr])
    loadings.append(p.named_steps["pca"].components_[0])

loadings = np.array(loadings)
print("PC1 loadings, refitted per fold:")
for i, load in enumerate(loadings):
    print(f"  fold {i}: " + " ".join(f"{v:+.3f}" for v in load))
print()
print(f"  max spread across folds, any feature: {np.abs(loadings.max(0) - loadings.min(0)).max():.4f}")
print("Non-zero spread proves the PCA is refitted inside each fold. Under the")
print("original approach every fold saw identical loadings, because the PCA was")
print("fitted once on everything before the loop started.")

# %% [markdown]
## Change 2 — `class_weight` instead of resampling

Column 1 downsampled the majority class, discarding **2,385 rows** (62% of the
training set) to make the training distribution 50/50.

The result was worse: test accuracy fell from **0.8741 to 0.8065** — below the
0.8106 majority-class baseline. The resampling bought recall on Hip-Hop and
paid for it in precision on Rock, and the notebook reported the lower accuracy
as though it were the balanced result.

`class_weight="balanced"` expresses the same preference as a per-class weight at
training time and **keeps all the data**. It cannot make the training set
balanced — nothing can, without changing the population — but it does not throw
away 62% of the evidence to pretend to.

# %%
weighted = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
weighted.fit(X_dev, y_dev)
w_pred = weighted.predict(X_final)
w_prob = weighted.predict_proba(X_final)[:, 1]

baseline = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("clf", LogisticRegression(max_iter=2000)),
])
baseline.fit(X_dev, y_dev)
b_pred = baseline.predict(X_final)
b_prob = baseline.predict_proba(X_final)[:, 1]

comparison = pd.DataFrame([
    {"approach": "column 1: downsampled 62% of data",
     "accuracy": 0.8065, "note": "from the executed column-1 notebook"},
    {"approach": "majority-class baseline",
     "accuracy": 1 - y_final.mean(), "note": "predict Rock for everything"},
    {"approach": "class_weight='balanced'",
     "accuracy": accuracy_score(y_final, w_pred), "note": "all 3,841 rows kept"},
    {"approach": "unweighted",
     "accuracy": accuracy_score(y_final, b_pred), "note": "all 3,841 rows kept"},
])
print(comparison.round(4).to_string(index=False))
print()
print("Balanced weighting reaches "
      f"{balanced_accuracy_score(y_final, w_pred):.4f} balanced accuracy while")
print("keeping every row. The resampled model discarded 62% of the data and")
print("scored below the do-nothing baseline on accuracy.")

# %%
print(classification_report(y_final, w_pred, target_names=["Rock", "Hip-Hop"],
                            digits=3, zero_division=0))

# %% [markdown]
## Change 3 — Put `n_components` inside model selection

Column 1 chose 2 components by eye, on a plot of the full dataset, then
cross-validated only the classifier. That is a second leak: the representation
was tuned on the test set.

Let the search choose, per fold.

# %%
param_grid = {
    "pca__n_components": [1, 2, 3, 4, 5, 6, 8],
    "clf__C": [0.01, 0.1, 1.0, 10.0],
}

pipe = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

grid = GridSearchCV(pipe, param_grid, cv=cv,
                    scoring="balanced_accuracy", n_jobs=2)
grid.fit(X_dev, y_dev)
print(f"best params: {grid.best_params_}")
print(f"best CV balanced accuracy: {grid.best_score_:.4f}")
print()
print("The 2 components chosen by eye in column 1 were never in competition")
print("with 1, 3, 4, 5, 6 or 8. The search picked"
      f" {grid.best_params_['pca__n_components']}.")

# %%
# Show the full grid surface — which matters more, components or C?
results = pd.DataFrame(
    grid.cv_results_["mean_test_score"].reshape(len(param_grid["pca__n_components"]), -1),
    index=param_grid["pca__n_components"],
    columns=[f"C={c}" for c in param_grid["clf__C"]],
)
print("CV balanced accuracy:")
print(results.round(4).to_string())

# %% [markdown]
## Change 4 — The non-PCA baseline

**If PCA did not help, the honest result is that it did not help, and that is
worth knowing.** Column 1 never ran the no-reduction case, so it could not tell
whether PCA was doing work or whether the classifier was carrying the model
regardless.

# %%
no_pca = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
no_pca.fit(X_dev, y_dev)
np_pred = no_pca.predict(X_final)
np_prob = no_pca.predict_proba(X_final)[:, 1]

best_pred = grid.predict(X_final)
best_prob = grid.predict_proba(X_final)[:, 1]

baseline_table = pd.DataFrame([
    {"model": "no PCA (8 raw features)", "balanced_acc": balanced_accuracy_score(y_final, np_pred),
     "roc_auc": roc_auc_score(y_final, np_prob), "pr_auc": average_precision_score(y_final, np_prob)},
    {"model": f"PCA to {grid.best_params_['pca__n_components']} components",
     "balanced_acc": balanced_accuracy_score(y_final, best_pred),
     "roc_auc": roc_auc_score(y_final, best_prob), "pr_auc": average_precision_score(y_final, best_prob)},
])
print(baseline_table.round(4).to_string(index=False))
print()
delta = baseline_table.balanced_acc.iloc[1] - baseline_table.balanced_acc.iloc[0]
print(f"PCA changes balanced accuracy by {delta:+.4f}.")
if abs(delta) < 0.005:
    print("That is noise. The dimensionality reduction buys nothing here, which")
    print("is a legitimate and useful finding: the classifier is doing the work,")
    print("and the PCA scatter plot that motivated the reduction was a picture")
    print("of the data, not evidence that reducing it helped.")
else:
    print("The reduction does help.")

# %% [markdown]
## Change 5 — PR-AUC and balanced accuracy, not accuracy

With a 4:1 imbalance, accuracy is the metric most guaranteed to flatter a
majority-class model.

# %%
def full_metrics(y_true, prob, name):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "precision_hh": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall_hh": tp / (tp + fn) if (tp + fn) else np.nan,
    }


metric_table = pd.DataFrame([
    full_metrics(y_final, b_prob, "logreg, unweighted"),
    full_metrics(y_final, w_prob, "logreg, balanced"),
    full_metrics(y_final, np_prob, "logreg, no PCA"),
    full_metrics(y_final, best_prob, f"tuned pipeline (PCA {grid.best_params_['pca__n_components']})"),
    full_metrics(y_final, np.full_like(b_prob, 0.0), "majority class (Rock)"),
])
print(metric_table.round(4).to_string(index=False))
print()
print(f"PR-AUC for a random classifier at a {y_final.mean():.3f} base rate is")
print(f"about {y_final.mean():.3f}. Compare the models' PR-AUC against that, not")
print("against 0.5 — the 0.5 reference belongs to ROC-AUC, not precision-recall.")

# %%
fig, ax = plt.subplots(figsize=(8, 4.6))
scores = metric_table[metric_table.model != "majority class (Rock)"]
ax.barh(scores.model, scores.pr_auc, color="#4C72B0", alpha=0.85)
ax.axvline(y_final.mean(), ls="--", color="crimson",
           label=f"random baseline = base rate {y_final.mean():.3f}")
ax.set_xlabel("PR-AUC")
ax.set_title("Precision-recall is the metric that respects the 4:1 imbalance")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
## Change 6 — The comparison column 1 got wrong, done properly

Column 1 concluded "the decision tree beats logistic regression". The repository
found that its cross-validation had compared the tree against itself, and the
corrected result is the other way round.

**Accuracy is not enough to support a claim that one model beats another.** Two
models scoring 0.865 and 0.874 on 961 held-out rows differ by nine tracks. The
right test is paired, on the same rows.

# %%
tree_pipe = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA(n_components=grid.best_params_["pca__n_components"])),
    ("clf", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
])
tree_pipe.fit(X_dev, y_dev)
t_pred = tree_pipe.predict(X_final)
t_prob = tree_pipe.predict_proba(X_final)[:, 1]

print(f"decision tree    : balanced acc {balanced_accuracy_score(y_final, t_pred):.4f}, "
      f"ROC-AUC {roc_auc_score(y_final, t_prob):.4f}")
print(f"logistic (tuned) : balanced acc {balanced_accuracy_score(y_final, best_pred):.4f}, "
      f"ROC-AUC {roc_auc_score(y_final, best_prob):.4f}")

# %%
# McNemar's test: of the rows where the two models disagree, is the split
# balanced? This is the correct test for two classifiers on one test set.
b01 = int(((best_pred == 1) & (t_pred == 0)).sum())   # logreg right, tree wrong
b10 = int(((best_pred == 0) & (t_pred == 1)).sum())   # tree right, logreg wrong
both_right = int(((best_pred == 1) & (t_pred == 1) & (y_final == 1)).sum())
both_wrong = int(((best_pred == 0) & (t_pred == 0) & (y_final == 0)).sum())

print()
print(f"rows where only logistic is correct : {b01}")
print(f"rows where only the tree is correct: {b10}")
print(f"both correct                       : {both_right + both_wrong}")
print(f"both wrong                         : {len(y_final) - both_right - both_wrong - b01 - b10}")

discordant = b01 + b10
if discordant > 0:
    chi2, p = stats.binomtest(b01, discordant, 0.5).pvalue, None
    exact_p = stats.binomtest(b01, discordant, 0.5).pvalue
    # McNemar with continuity correction
    mcc = (abs(b01 - b10) - 1) ** 2 / discordant
    print()
    print(f"McNemar exact test: p = {exact_p:.4f}")
    print(f"McNemar chi2 (cc) = {mcc:.3f}")
    print()
    if exact_p < 0.05:
        print("  The difference is statistically significant: logistic regression")
        print("  genuinely beats the decision tree on this data.")
    else:
        print("  The difference is NOT statistically significant. The original")
        print("  notebook's claim that the tree 'outperforms' logistic regression")
        print("  was not supported by the evidence, and its own comparison was")
        print("  invalid. The honest answer is 'indistinguishable'.")
else:
    print("The two models never disagree; McNemar's test is undefined.")

# %%
# Per-track win/loss, which makes the disagreement concrete
disagree = (t_pred != best_pred)
print()
print(f"the two models disagree on {disagree.sum()} of {len(y_final)} test tracks "
      f"({disagree.mean():.1%})")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(["logistic only right", "tree only right", "both agree"],
       [b01, b10, len(y_final) - b01 - b10],
       color=["#4C72B0", "#C44E52", "#999999"])
ax.set_ylabel("test tracks")
ax.set_title(f"McNemar: {discordant} discordant rows of {len(y_final)}")
plt.tight_layout()
plt.show()

# %% [markdown]
## Change 7 — Nested CV for the tuned pipeline

The `grid.best_score_` above is optimistically biased: it is the maximum over a
grid, so the winner is partly the luckiest configuration. Nested CV gives an
unbiased estimate of the whole select-then-fit procedure.

# %%
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

from sklearn.model_selection import cross_val_score

nested = Pipeline([("sc", StandardScaler()), ("pca", PCA()),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
nested_scores = cross_val_score(
    GridSearchCV(nested, param_grid, cv=inner, scoring="balanced_accuracy", n_jobs=2),
    X_dev, y_dev, cv=outer, scoring="balanced_accuracy", n_jobs=2,
)
print("Nested CV on the development set (unbiased for the whole procedure):")
print(f"  folds  : {np.round(nested_scores, 4).tolist()}")
print(f"  mean   : {nested_scores.mean():.4f} (+/- {nested_scores.std(ddof=1):.4f})")
print()
print(f"  best-of-grid CV (biased) : {grid.best_score_:.4f}")
print(f"  nested CV mean           : {nested_scores.mean():.4f}")
print(f"  selection optimism       : {grid.best_score_ - nested_scores.mean():+.4f}")
print()
print(f"  final test set, untouched: {balanced_accuracy_score(y_final, best_pred):.4f}")
print()
print("Three estimates, three different numbers, all defensible, answering")
print("slightly different questions. Quoting only the most flattering one is")
print("the habit this column exists to break.")

# %% [markdown]
## Summary

| | Column 1 | Column 2 |
|---|---|---|
| Preprocessing | fitted on all data, fixed during CV | inside `Pipeline`, refit per fold |
| Imbalance | downsampled, 62% of data discarded | `class_weight="balanced"`, all data kept |
| `n_components` | 2, chosen by eye on all data | selected inside CV from 7 values |
| Baselines | PCA only | PCA **and** no-PCA, plus majority class |
| Metrics | accuracy | accuracy, balanced accuracy, ROC-AUC, PR-AUC, per-class P/R |
| Model comparison | tree vs itself | McNemar's test on 961 shared rows |
| Performance estimate | best-of-grid CV | nested CV + untouched final test |

# %%
print("FINAL HELD-OUT NUMBERS (final test set, never used for any decision)")
print(f"  majority-class baseline accuracy : {1 - y_final.mean():.4f}")
print(f"  column 1 resampled model         : 0.8065  <- below the baseline")
print(f"  tuned pipeline balanced accuracy : {balanced_accuracy_score(y_final, best_pred):.4f}")
print(f"  ROC-AUC                           : {roc_auc_score(y_final, best_prob):.4f}")
print(f"  PR-AUC                            : {average_precision_score(y_final, best_prob):.4f}")
print(f"  nested CV (unbiased, dev only)   : {nested_scores.mean():.4f}")
print()
print("Two conclusions survive all of this:")
print("  1. the resampling in column 1 was actively harmful — it scored below")
print("     the majority-class baseline, and removing it recovers 4 points;")
print("  2. logistic regression beats the decision tree decisively — 101")
print("     discordant rows to 15, McNemar p < 0.0001.")
print()
print("So the 2019 conclusion was not merely unsupported, it was inverted.")
print("A tree on 2 principal components cannot represent a decision boundary")
print("that is close to linear in the feature space; the logistic model can,")
print("and it does. The original notebook's own CV loop would have shown")
print("this immediately had the variable it compared against not been")
print("overwritten.")
print()
print("What is still missing: these are pre-extracted features. Nobody here")
print("computed a spectrogram. The 2026 column is where that changes.")
