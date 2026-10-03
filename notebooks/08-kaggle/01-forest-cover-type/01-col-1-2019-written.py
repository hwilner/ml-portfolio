# 1 — Forest Cover Type: ExtraTrees + AdaBoost + GB + HistGB (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The 2019 Kaggle kernel `hareljwil/cover`,
ported to run outside Kaggle with minimal changes so the original reasoning is
preserved and the later columns have something honest to improve on.

## The original, and what it did

Fifty-four covariates describing 30×30m cells of terrain (elevation, slope,
aspect, distance to hydrology/roads/fire points, soil and wilderness one-hot
blocks). Seven `Cover_Type` classes, imbalanced. The kernel:

1. EDA — null counts, `describe()`, a correlation matrix
2. Collapsed the 40 soil one-hot columns and the 4 wilderness one-hot columns
   to a single `argmax` integer each
3. Invented two features: `shade = Hillshade_9am * Hillshade_Noon²`, and
   `distance_hyd` (which is computed and then never used)
4. Kept **5 columns out of 54**: two distance features plus the three above
5. Fit a `VotingClassifier` over ExtraTrees, AdaBoost, GradientBoosting and
   HistGradientBoosting at default settings
6. One `train_test_split` at 40% test, `random_state=0`, **not stratified**
7. Reported `clf.score()` — plain accuracy

**Correction, made after running it:** the notebook above calls this an
"imbalanced" problem, and it is not. `Cover_Type` is exactly balanced — 2,160
observations in each of 7 classes, so the majority-class baseline is 1/7 =
0.1429. Stratification therefore changes almost nothing, and column 2's case
for fixing the *evaluation* rests on the single split and the missing baseline,
not on class imbalance. That was wrong before this cell ran.

The three columns then fix, respectively: the evaluation, the feature
engineering, and the tooling.

## Data

`forest-cover-type-kernels-only`, 15,120 training rows. Fetched, not committed:

```bash
kaggle competitions download -c forest-cover-type-kernels-only \
    -p data/kaggle/forest-cover-type-kernels-only
```

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.kaggle_data import competition_path

SEED = 20191211
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

train = pd.read_csv(competition_path("forest-cover-type-kernels-only", "train.csv"))
print(f"train: {train.shape}")
train.isnull().sum()  # the 2019 null check — kept because it is genuinely useful

# %%
train.describe()

# %%
# 2019: plt.matshow(train.corr())
mat = train.corr()
print(f"correlation matrix: {mat.shape}")

# %%
plt.figure(figsize=(9, 8))
sns.heatmap(mat, cmap="viridis", center=0, square=True, cbar_kws={"shrink": 0.6})
plt.title("Forest Cover Type: feature correlation")
plt.tight_layout()
plt.show()

# %%
# The original selected the one-hot blocks by POSITION: col[15:-1] for soil and
# col[11:15] for wilderness. That works only for this exact column order, so
# the names are resolved by prefix instead and the result is checked against
# the original's assumption.
col = list(train)
soil = train[col[15:-1]].values
wild = train[col[11:15]].values
print(f"positional: {len(soil[0])} soil columns, {len(wild[0])} wilderness columns")

soil_cols = [c for c in train.columns if c.startswith("Soil")]
wild_cols = [c for c in train.columns if c.startswith("Wilderness")]
print(f"by name   : {len(soil_cols)} soil columns, {len(wild_cols)} wilderness columns")
assert len(soil_cols) == soil.shape[1] and len(wild_cols) == wild.shape[1]
print("positional and by-name selections agree")

# %%
soil_cat = np.argmax(soil, axis=1)
wild_cat = np.argmax(wild, axis=1)
distance_hyd = (train["Vertical_Distance_To_Hydrology"].values ** 2
                + train["Horizontal_Distance_To_Hydrology"].values ** 2) ** 0.5
shade = (train["Hillshade_9am"].values
         * train["Hillshade_Noon"].values * train["Hillshade_Noon"].values)

# %%
# The 2019 kernel used 5 of 54 columns. Note that distance_hyd is computed
# above and never reaches the feature matrix — it was dead code.
X = train.loc[:, ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]].copy()
X["shade"] = shade
X["wild_cat"] = wild_cat
X["sioil_cat"] = soil_cat  # 2019 spelling, kept so column 3 can report the typo
y = train["Cover_Type"]

print(f"features used: {X.shape[1]} of {train.shape[1]}")
print(f"discarded    : {train.shape[1] - X.shape[1]}")
print(f"\nclass distribution:\n{y.value_counts().sort_index().to_string()}")
maj = y.value_counts(normalize=True).max()
print(f"\nmajority-class accuracy = {maj:.4f}")
print(f"imbalance ratio (max/min class) = {y.value_counts().max() / y.value_counts().min():.2f}")
print("\nNote: this dataset is BALANCED, 2160 per class. An earlier draft of this\n"
      "notebook described the problem as imbalanced, which the data above\n"
      "contradicts, and the description was corrected rather than kept.")

# %%
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.model_selection import train_test_split

clf = VotingClassifier(
    [("cl1", ExtraTreesClassifier()),
     ("cl2", AdaBoostClassifier()),
     ("cl3", GradientBoostingClassifier()),
     ("cl4", HistGradientBoostingClassifier())],
    voting="hard")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
clf.fit(X_train, y_train)

# %%
# The 2019 result. Accuracy on a single unstratified 40% split.
accuracy = clf.score(X_test, y_test)
print(f"clf.score() on the 2019 split = {accuracy:.4f}")
print(f"majority class                = {y.value_counts(normalize=True).max():.4f}")
print(f"\nlift over majority = {accuracy - y.value_counts(normalize=True).max():+.4f}")

# %%
test = pd.read_csv(competition_path("forest-cover-type-kernels-only", "test.csv"))
ID = test["Id"]
soil = test[col[15:-1]].values
wild = test[col[11:15]].values
soil_cat = np.argmax(soil, axis=1)
wild_cat = np.argmax(wild, axis=1)
shade = test["Hillshade_9am"].values * test["Hillshade_Noon"].values ** 2
X_sub = test.loc[:, ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]].copy()
X_sub["shade"] = shade
X_sub["wild_cat"] = wild_cat
X_sub["sioil_cat"] = soil_cat

predictions = clf.predict(X_sub)
sub = pd.DataFrame({"Id": ID, "Cover_Type": predictions})
print(f"submission: {sub.shape}")
print(f"predicted class distribution:\n{sub.Cover_Type.value_counts().sort_index().to_string()}")
sub.to_csv("submission.csv", index=False)
print("\nwrote submission.csv")

# %%
# 2019 SUMMARY
# %%
print(f"""
FOREST COVER TYPE — COLUMN 1 (2019, AS WRITTEN)

Model    VotingClassifier (ExtraTrees + AdaBoost + GradientBoosting +
                          HistGradientBoosting), all defaults
Features 5 of 54: two distance covariates, a squaring product, and two
         argmax-collapsed one-hot blocks
Split    one 40% test split, random_state=0, NOT stratified
Metric   accuracy

Measured accuracy {accuracy:.4f} against a {maj:.4f} majority baseline.

The four things columns 2 and 3 address:
  1. accuracy from a SINGLE unstratified 40% split, with no confidence
     interval and no baseline quoted alongside it
     (the class distribution is balanced, so stratification matters less
     than the header of this notebook once claimed)
  2. 49 of 54 covariates discarded, and a soil one-hot block of 40 columns
     collapsed to an argmax integer, which throws away every degree of
     confidence the encoding contained
  3. soil and wilderness selected by column position, not by name
  4. a column named sioil_cat, a typo that survived into the submission
""")
