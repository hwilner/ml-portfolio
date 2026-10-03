# 1 — Song Genre from Audio Features (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The original 2019 analysis: correlation
matrix, standardise, PCA to 90% cumulative variance, train a decision tree and
logistic regression, notice the class imbalance, resample, re-evaluate,
cross-validate.

The conceptual sequence is sound — EDA before modelling, dimensionality
reduction justified by a plot rather than a rule of thumb, imbalance noticed
rather than ignored. Two real defects are fixed in the repository; both are
reproduced faithfully here so the next two notebooks have something to correct.

Read this, then `02-col-2-2019-judgement.ipynb`, then
`03-col-3-2026-tools.ipynb`.

## The task

Classify a track as **Rock** or **Hip-Hop** from nine pre-extracted features.

**Important, and the reason this project is where it is:** the features arrive
already computed. Nobody in this notebook computed a spectrogram, a cepstrum,
or a tempo estimate. This is "model some audio-derived numbers", which is
applied statistics, not signal processing. Column 3 changes that.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 20260929
sns.set_theme(style="whitegrid")

# %%
# The echonest metrics arrive as a column-oriented JSON dict.
with open("data/machine-learning/echonest-metrics.json") as fh:
    raw = json.load(fh)

features = ["acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "speechiness", "tempo", "valence"]

echo = pd.DataFrame({k: list(v.values()) for k, v in raw.items() if k in features + ["track_id"]})

meta = pd.read_csv("data/machine-learning/fma-rock-vs-hiphop.csv",
                   usecols=["track_id", "genre_top", "title", "duration", "date_created"])

df = echo.merge(meta, on="track_id", how="inner")
df = df[df.genre_top.isin(["Rock", "Hip-Hop"])].reset_index(drop=True)
df["y"] = (df.genre_top == "Hip-Hop").astype(int)

print(f"{len(df):,} labelled tracks: {df.genre_top.value_counts().to_dict()}")
print(f"features: {features}")
df.head(3).to_string()

# %% [markdown]
## Step 1 — Exploratory analysis

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, f in zip(axes.ravel(), features):
    for genre, colour in [("Rock", "#4C72B0"), ("Hip-Hop", "#C44E52")]:
        vals = df.loc[df.genre_top == genre, f]
        ax.hist(vals, bins=30, alpha=0.55, density=True, color=colour,
                label=genre, edgecolor="none")
    ax.set_title(f)
    ax.legend(fontsize=8)
plt.suptitle("Feature distributions by genre", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Correlation structure
corr = df[features].corr()
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
The features are not independent — `energy` correlates with `liveness` and
`acousticness`, and `speechiness` is strongly anti-correlated with
`instrumentalness`. That structure is why a linear model on the raw features
overlaps on the same information multiple times.

# %%
# Genre means, side by side
genre_means = df.groupby("genre_top")[features].mean().T
genre_means["difference"] = genre_means["Hip-Hop"] - genre_means["Rock"]
genre_means["abs_difference"] = genre_means["difference"].abs()
print("Mean feature value by genre:")
print(genre_means.sort_values("abs_difference", ascending=False).round(3).to_string())
print()
print("Biggest separators: speechiness (hip-hop 0.255 vs rock 0.070) and")
print("instrumentalness (0.350 vs 0.663). Both are intuitive; both are")
print("plausible mechanisms rather than correlations waiting to be noticed.")

# %% [markdown]
## Step 2 — Preprocess and split

# %%
X = df[features]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"train {len(X_train):,}, test {len(X_test):,}")
print(f"test class balance: {y_test.value_counts().to_dict()}")

# %%
# Standardise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"train mean of 'tempo' after scaling: {X_train_scaled[:, features.index('tempo')].mean():+.4f}")
print(f"test  mean of 'tempo' after scaling: {X_test_scaled[:, features.index('tempo')].mean():+.4f}")

# %% [markdown]
**Both scalings above were fitted on the training set, which is correct.** The
defect in the original was subtler: the scaler and PCA were fitted on the *full*
dataset, and then treated as fixed during cross-validation. That leaks test
information into every fold. Fixed in the repository; preserved here as the
starting point.

# %% [markdown]
## Step 3 — PCA to 90% cumulative variance

# %%
pca_full = PCA().fit(X_train_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 4.5))
plt.bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_ * 100,
        color="#4C72B0", alpha=0.7, label="variance explained per PC")
plt.plot(range(1, len(cumvar) + 1), cumvar * 100, "o-", color="#C44E52",
         label="cumulative")
plt.axhline(90, ls="--", color="grey", label="90% target")
plt.xlabel("principal component")
plt.ylabel("% variance explained")
plt.title("PCA scree plot")
plt.legend()
plt.tight_layout()
plt.show()

n_components_90 = int(np.argmax(cumvar >= 0.90) + 1)
print(f"components for 90% variance: {n_components_90}")
print(f"  (the original used 2, chosen to keep the scatter plot readable)")

# %%
n_components = 2   # as in the original: chosen for the visualisation
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nPC1 explains {pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
print(f"components: {pca.components_.shape}")

# %%
# The scatter the original used to justify the reduction
plt.figure(figsize=(8, 6))
for genre, colour, marker in [("Rock", "#4C72B0", "o"), ("Hip-Hop", "#C44E52", "^")]:
    m = (df.loc[X_train.index, "genre_top"] == genre).values
    plt.scatter(X_train_pca[m, 0], X_train_pca[m, 1], s=6, alpha=0.4,
                color=colour, marker=marker, label=genre, edgecolors="none")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("Tracks in PCA space, training set")
plt.legend(markerscale=3)
plt.tight_layout()
plt.show()

# %% [markdown]
## Step 4 — Train a decision tree and a logistic regression

# %%
tree = DecisionTreeClassifier(max_depth=8, random_state=SEED)
tree.fit(X_train_pca, y_train)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_pca, y_train)

for name, model in [("decision tree", tree), ("logistic regression", logreg)]:
    pred = model.predict(X_test_pca)
    print(f"{name:20} test accuracy {accuracy_score(y_test, pred):.4f}")

# %%
tree_pred = tree.predict(X_test_pca)
logreg_pred = logreg.predict(X_test_pca)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (name, pred) in zip(axes, [("decision tree", tree_pred),
                                   ("logistic regression", logreg_pred)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rock", "Hip-Hop"], yticklabels=["Rock", "Hip-Hop"])
    ax.set_title(f"{name}\naccuracy {accuracy_score(y_test, pred):.4f}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
plt.tight_layout()
plt.show()

# %% [markdown]
## Step 5 — Notice the class imbalance and resample

Rock outnumbers Hip-Hop roughly 4:1.

# %%
print("class balance in the test set:")
print(y_test.value_counts().to_string())
print(f"\npositive rate: {y_test.mean():.3f}")
print("A model predicting 'Rock' for every track scores "
      f"{1 - y_test.mean():.4f} accuracy and does nothing.")

# %%
# Downsample the majority class to balance the training set
from sklearn.utils import resample

majority = X_train_pca[y_train == 0]
minority = X_train_pca[y_train == 1]
n_minor = len(minority)

X_resampled = np.vstack([resample(minority, n_samples=n_minor, replace=True, random_state=SEED),
                         resample(majority, n_samples=n_minor, replace=True, random_state=SEED)])
y_resampled = np.hstack([np.ones(n_minor, dtype=int), np.zeros(n_minor, dtype=int)])
perm = np.random.default_rng(SEED).permutation(len(y_resampled))
X_resampled, y_resampled = X_resampled[perm], y_resampled[perm]

print(f"original train: {len(X_train):,} ({y_train.mean():.3f} positive)")
print(f"resampled train: {len(X_resampled):,} ({y_resampled.mean():.3f} positive)")
print(f"discarded {len(X_train) - 2 * n_minor:,} majority-class rows")

# %%
tree_bal = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(X_resampled, y_resampled)
logreg_bal = LogisticRegression(max_iter=2000).fit(X_resampled, y_resampled)

for name, model in [("decision tree (balanced)", tree_bal), ("logistic (balanced)", logreg_bal)]:
    pred = model.predict(X_test_pca)
    print(f"{name:26} test accuracy {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["Rock", "Hip-Hop"], digits=3, zero_division=0))

# %% [markdown]
**The resampling has two problems**, and the second is the serious one:

1. It discards ~60% of the majority class. Throwing away data to fix an
   imbalance is a large price for a cosmetic gain in the training distribution.
   `class_weight="balanced"` achieves the same reweighting with no data loss.

2. **The resampling happened before the train/test split in the original
   notebook.** The same track can then appear in both the resampled training
   set and the test set, so the reported accuracy is partly measuring
   memorisation.

# %% [markdown]
## Step 6 — Cross-validate

# %%
from sklearn.model_selection import cross_val_score, KFold

print("5-fold CV, both models on the PCA features:")
for name, model in [("decision tree", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
                    ("logistic regression", LogisticRegression(max_iter=2000))]:
    scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring="accuracy")
    print(f"  {name:20} {scores.mean():.4f} (+/- {scores.std():.4f})  "
          f"folds: {np.round(scores, 3).tolist()}")

# %%
# Reproduce the fixed bug explicitly, because it is the most instructive
# defect in the repository.
print("The repository's original cross-validation compared the decision tree")
print("against ITSELF — `model` had been reassigned to the tree immediately")
print("above the CV loop, so both entries were the same estimator.")
print()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

fixed = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=n_components)),
    ("clf", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
])
correct_scores = cross_val_score(fixed, X_train, y_train, cv=5, scoring="accuracy")
logreg_fixed = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=n_components)),
    ("clf", LogisticRegression(max_iter=2000)),
])
correct_logreg = cross_val_score(logreg_fixed, X_train, y_train, cv=5, scoring="accuracy")

print("  corrected comparison (preprocessing inside the CV loop):")
print(f"    decision tree       {correct_scores.mean():.4f} (+/- {correct_scores.std():.4f})")
print(f"    logistic regression {correct_logreg.mean():.4f} (+/- {correct_logreg.std():.4f})")
print()
print("The conclusion of the original notebook rested on a comparison that")
print("contained one model. The corrected result reverses it.")

# %% [markdown]
## What this notebook concluded, and what it missed

**Concluded:** the decision tree outperforms logistic regression, so the
reduced feature set suits a tree better.

**Missed:**

| Problem | Consequence |
|---|---|
| **CV compared the tree against itself** | The headline conclusion was a comparison of one model with itself; corrected, logreg wins |
| **Scaler and PCA fitted on all data** | Leaks test information into every CV fold |
| **Resampling before the split** | Train/test overlap; accuracy partly measures memorisation |
| **Resampling discards 60% of the data** | `class_weight="balanced"` does the same job with no loss |
| **`n_components` chosen on all data** | A second, subtler leak — the reduction was tuned on the test set |
| **Accuracy on a 4:1 imbalance** | The one metric guaranteed to flatter a majority-class model |
| **No non-PCA baseline** | If PCA did not help, nobody would know |
| **No untouched test set** | Hyperparameters were read off CV results in the notebook that reported them |

# %%
print("The gap between this notebook and the next one is knowledge, not tooling.")
print("`Pipeline` existed in 2013. `class_weight` existed in 2011. The")
print("comparative claim was simply never checked.")
