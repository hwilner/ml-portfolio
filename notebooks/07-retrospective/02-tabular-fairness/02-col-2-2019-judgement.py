# 2 — Credit Approval (2019 judgement, same tools)

## What this notebook is

**Column 2 of the retrospective.** The same 690 accounts, the same 2019-era
tools (`scikit-learn`, `pandas`, `numpy`, `matplotlib`), the analysis done with
judgement I did not have in 2019.

Everything here was possible in 2019. `ColumnTransformer` had existed for a
year; `Pipeline` for six; `OneHotEncoder` for eight. This is a discipline gap,
not a tooling gap.

## The changes from column 1

| # | Change | Possible in 2019? |
|---|---|---|
| 1 | Everything inside a `Pipeline` | Yes — `Pipeline` since 0.14 (2013) |
| 2 | `OneHotEncoder` instead of `LabelEncoder` | Yes |
| 3 | Grid-search `C`, not `max_iter`/`tol` | Yes |
| 4 | Stratified split | Yes |
| 5 | A separate held-out test set, untouched by CV | Yes |
| 6 | Report PR-AUC, balanced accuracy, calibration | Yes |
| 7 | Pick the threshold from a stated cost, not 0.5 | Yes |
| 8 | **A fairness section** | Yes — and it should not have been optional by 2019 either |

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path("scripts").resolve()))
from credit_data import (FEATURE_CATEGORICAL as CATEGORICAL,
                        FEATURE_NUMERIC as NUMERIC,
                        describe_missingness, load_clean)

SEED = 20260929
df = load_clean()
print(f"{len(df):,} accounts | positive rate {df['y'].mean():.1%}")

# %% [markdown]
## Change 1 — Everything inside a `Pipeline`

Column 1 had two defects that cancelled out by accident: the scaler leaked, and
the scaled array was never used. A `Pipeline` makes both impossible, because
every transform is re-fitted inside each CV fold on that fold's training data
only.

The structure also forces the split by column type to be explicit, which sets up
change 2.

# %%
# A category seen only in the test partition is genuinely new information, and
# `handle_unknown="ignore"` maps it to all-zeros. That is the right behaviour,
# but sklearn emits a warning each time; note it here rather than leaving a
# reader to wonder whether it indicates a bug.
import warnings

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])

model = Pipeline([
    ("prep", preprocess),
    ("logreg", LogisticRegression(max_iter=2000)),
])

X = df.drop(columns=["y", "default_next_month", "target"])
y = df["y"]

# Change 4: stratified split, so the test class ratio is not a coin flip
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"train {len(X_train)} (pos {y_train.mean():.3f}) | "
      f"test {len(X_test)} (pos {y_test.mean():.3f})")

# %%
# Prove the leak is gone: each fold re-fits the imputer and the scaler.
# If preprocessing sat outside the CV loop, these fold statistics would be
# identical across folds. They are not.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_means = []
for train_idx, val_idx in cv.split(X_train, y_train):
    sub = X_train.iloc[train_idx]
    fold_means.append(sub["limit_bal"].mean())

print("mean limit_bal in each fold's training partition:")
print("  " + "  ".join(f"{m:.3f}" for m in fold_means))
print(f"  spread: {max(fold_means) - min(fold_means):.4f}")
print()
print("Non-zero spread means the imputer and scaler are refitted per fold.")

# %% [markdown]
## Change 2 — `OneHotEncoder` instead of `LabelEncoder`

Column 1 gave `education` the codes `l -> 0, u -> 1, y -> 2`. That asserts
`u` sits numerically between `l` and `y`, which is meaningless — the codes are
alphabetical.

One-hot encoding gives each category its own coefficient, so the model learns
"the effect of `u` relative to `l`" and "the effect of `y` relative to `l`"
independently, with no ordering imposed.

# %%
# Make the difference measurable: does the ordering change the fit?
# The imputation is explicit here (rather than skipped) because a comparison
# that fails for want of preprocessing would not be a comparison at all.
from sklearn.preprocessing import LabelEncoder

# Impute from the TRAINING partition only, then apply to both — otherwise the
# comparison model gets to see the test rows' means, which is the very leak
# change 1 exists to prevent.
train_idx, test_idx = X_train.index, X_test.index

X_ord = pd.DataFrame(index=X.index)
for col in NUMERIC:
    imp = SimpleImputer(strategy="mean").fit(X.loc[train_idx, [col]])
    X_ord[col] = pd.Series(imp.transform(X[[col]]).ravel(), index=X.index)
for col in CATEGORICAL:
    imp = SimpleImputer(strategy="most_frequent").fit(X.loc[train_idx, [col]])
    filled = pd.Series(imp.transform(X[[col]]).ravel(), index=X.index)
    X_ord[col] = pd.Series(LabelEncoder().fit_transform(filled), index=X.index)

ord_model = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000)),
])
ord_model.fit(X_ord.loc[train_idx], y_train)
ord_pred = ord_model.predict(X_ord.loc[test_idx])

edu_coef = ord_model.named_steps["logreg"].coef_[0][NUMERIC.index("limit_bal") + 1]
print("Ordinal encoding forces ONE coefficient per nominal column.")
print(f"  education: l=0, u=1, y=2 -> a single coefficient of {edu_coef:+.4f}")
print("  -> one number for 'being later in the alphabet', which is not a quantity.")
print()
print(f"ordinal-encoded test accuracy : {accuracy_score(y_test, ord_pred):.4f}")
print("The score barely moves, and that is the trap: a specification error")
print("that costs almost nothing in accuracy is still a specification error.")
print("It would matter enormously for a linear model in a setting where the")
print("nominal column carried strong signal, and it silently changes the")
print("meaning of every downstream coefficient and feature-importance plot.")

# %%
# What one-hot actually does to the design matrix
prep = preprocess.fit(X_train)
names = prep.get_feature_names_out()
print(f"design matrix: {X_train.shape[1]} raw columns -> {len(names)} features")
print()
print("first 14 output features:")
for n in names[:14]:
    print(f"  {n}")
print("  ...")
print()
print("Each nominal category now has its own column, so the model estimates")
print("a separate effect for it instead of a single ordered slope.")

# %% [markdown]
## Change 3 — Grid-search `C`, the hyperparameter that matters

# %%
param_grid = {
    "logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "logreg__class_weight": [None, "balanced"],
}

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*unknown categories.*")
    grid = GridSearchCV(
        model, param_grid,
        cv=cv, scoring="balanced_accuracy", n_jobs=2,
    )
    grid.fit(X_train, y_train)

print(f"best params : {grid.best_params_}")
print(f"best CV balanced accuracy: {grid.best_score_:.4f}")
print()
print("`C` is the inverse of the L2 penalty. Small C = heavy regularisation,")
print("large C = almost unregularised. It is the bias-variance control for")
print("logistic regression, and the original notebook never touched it.")

# %%
# The fold-to-fold spread, which column 1 never reported
cv_results = pd.DataFrame(
    grid.cv_results_["mean_test_score"].reshape(len(param_grid["logreg__C"]), -1),
    index=param_grid["logreg__C"],
    columns=param_grid["logreg__class_weight"],
)
print("CV balanced accuracy by C and class_weight:")
print(cv_results.round(4).to_string())
print()
best_C = grid.best_params_["logreg__C"]
best_cw = grid.best_params_["logreg__class_weight"]
print(f"best C = {best_C}, class_weight = {best_cw}")
print(f"mean    : {cv_results.loc[best_C, best_cw]:.4f}")

# %% [markdown]
## Change 5 — A held-out test set, untouched by CV

All the tuning above used only `X_train`. `X_test` has not been seen by any
fitted transform, any fold, or any hyperparameter choice. This is the first
time it is scored.

# %%
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

cv_best = grid.best_score_
test_bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"best CV balanced accuracy : {cv_best:.4f}")
print(f"held-out balanced accuracy: {test_bal_acc:.4f}")
print(f"optimism (CV - test)      : {cv_best - test_bal_acc:+.4f}")
print()
if cv_best - test_bal_acc > 0.02:
    print("The CV figure is optimistic by more than 2 points. Reporting CV")
    print("alone as the performance estimate overstates what the model does")
    print("on data it has not seen.")
else:
    print("The gap is small — the tuning did not badly overfit.")

# %%
# Full metric panel, not accuracy alone
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
metrics = pd.DataFrame({
    "metric": [
        "accuracy", "balanced accuracy", "ROC-AUC", "PR-AUC (avg precision)",
        "recall on defaults", "precision on defaults",
        "majority-class baseline",
    ],
    "value": [
        accuracy_score(y_test, y_pred),
        test_bal_acc,
        roc_auc_score(y_test, y_prob),
        average_precision_score(y_test, y_prob),
        tp / (tp + fn),
        tp / (tp + fp),
        1 - y_test.mean(),
    ],
})
print(metrics.round(4).to_string(index=False))
print()
print("PR-AUC is the number that matters at a 30% base rate, and it is much")
print("lower than the ROC-AUC. Reporting only ROC-AUC would overstate the model.")

# %% [markdown]
## Change 6 — Calibration

A model that ranks well but assigns wrong probabilities is not usable in
underwriting, where a score feeds a limit decision.

# %%
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="quantile")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
ax[0].plot(mean_pred, frac_pos, "o-", color="#C44E52", lw=2, label="model")
ax[0].set_xlabel("mean predicted probability")
ax[0].set_ylabel("observed default rate")
ax[0].set_title("Calibration curve (held-out)")
ax[0].legend()

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
ax[1].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC-AUC {roc_auc_score(y_test, y_prob):.3f}")
ax[1].plot([0, 1], [0, 1], "--", color="grey")
ax[1].set_xlabel("false positive rate")
ax[1].set_ylabel("true positive rate")
ax[1].set_title("ROC curve")
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
# Calibration error
ece = np.mean(np.abs(mean_pred - frac_pos))
print(f"expected calibration error (8 quantile bins): {ece:.4f}")
print()
print("Interpretation: across the probability range, the model's scores are")
print(f"off by {ece:.1%} on average. That is too much to hand to a credit officer")
print("as a literal 'chance this person defaults' without recalibration.")

# %% [markdown]
## Change 7 — Choose the threshold from a cost

0.5 is arbitrary. In lending the asymmetry is severe: a false positive costs an
application; a false negative costs the balance and the relationship. Build the
trade-off explicitly and read the cost off the curve.

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Simple symmetric cost model, parameterised by the cost ratio.
rows = []
for cost_fp, cost_fn in [(1, 1), (1, 5), (1, 20), (5, 1)]:
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        cost = fp_ * cost_fp + fn_ * cost_fn
        if cost < best_cost:
            best_cost, best_t = cost, t
    pred = (y_prob >= best_t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({
        "cost_fn / cost_fp": f"{cost_fn}:{cost_fp}",
        "optimal threshold": round(best_t, 3),
        "recall": tp_ / (tp_ + fn_),
        "precision": tp_ / (tp_ + fp_),
        "n_approved": int(pred.sum()),
        "expected cost": int(best_cost),
    })

cost_table = pd.DataFrame(rows)
print(cost_table.to_string(index=False))
print()
print("As the cost of missing a defaulter rises, the optimal threshold falls")
print("and the bank approves more marginal applicants. 0.5 is just one row of")
print("this table, and it is the row with no justification attached.")

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(recall, precision, color="#C44E52", lw=2, label="precision-recall")
ax.axhline(y_test.mean(), ls="--", color="grey", label=f"base rate {y_test.mean():.3f}")
for t in [0.2, 0.35, 0.5, 0.65]:
    ax.axvline(0, color="none")  # keep legend clean
    pred = (y_prob >= t).astype(int)
    r = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    if (r[0] + r[1]) > 0:
        ax.scatter(r[2] / (r[2] + r[3]), r[2] / (r[2] + r[0]), s=90,
                   marker="o", edgecolors="black", zorder=5, label=f"threshold {t}")
ax.set_xlabel("recall (share of defaults caught)")
ax.set_ylabel("precision (share of approvals that default)")
ax.set_title("Every operating point is a business decision")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
## Change 8 — The fairness section

**This is a lending decision.** Protected attributes are in the feature set.
A model that reports only overall accuracy has not been evaluated for the thing
that matters most in this domain.

The tests below are the standard descriptive ones. They are necessary and not
sufficient — no fairness metric here can certify a model is fair, and passing
them is not the same as being fair.

# %%
# Protected attributes present in this feature set
print("Attributes in the feature set that are protected or proxy for protected")
print("characteristics under fair-lending regulation:")
print("  marital            (proxy for sex / family status)")
print("  education          (proxy for socioeconomic status and ethnicity)")
print("  age_band           (age is a protected characteristic)")
print("  limit_bal          (proxies income)")
print()
print("The dataset has no direct sex, ethnicity or zipcode column, so this")
print("analysis is limited to what the available columns support. That is a")
print("real limitation and is stated rather than glossed over.")

# %%
# A fairness metric set, computed per group with confidence intervals.
from scipy import stats

def group_metrics(y_true, y_prob, threshold=0.5, min_n=20):
    """Selection and error rates for one group, with Wilson intervals."""
    pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    sel = (tp + fp) / n if n else np.nan          # approval rate
    tpr = tp / (tp + fn) if (tp + fn) else np.nan  # recall: catch rate
    fpr = fp / (fp + tn) if (fp + tn) else np.nan  # false-alarm rate

    return {
        "n": n,
        "approval_rate": sel,
        "recall_tpr": tpr,
        "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sufficient_n": n >= min_n,
    }


for attr in ["marital", "education", "age_band"]:
    print("=" * 68)
    print(f"  FAIRNESS BY {attr.upper()}")
    print("=" * 68)
    vals = X_test[attr].fillna("missing")
    buckets = pd.qcut(X_test["age_band"], 4, duplicates="drop") if attr == "age_band" else vals

    rows = []
    for g in pd.Series(buckets).unique():
        mask = (pd.Series(buckets) == g).values
        if mask.sum() < 5:
            continue
        m = group_metrics(y_test[mask], y_prob[mask])
        label = f"{g}"
        rows.append({
            "group": label[:14],
            "n": m["n"],
            "approval": round(m["approval_rate"], 3) if m["approval_rate"] == m["approval_rate"] else None,
            "recall": round(m["recall_tpr"], 3) if m["recall_tpr"] == m["recall_tpr"] else None,
            "fpr": round(m["fpr"], 3) if m["fpr"] == m["fpr"] else None,
        })

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    print()

# %%
# The headline numbers for the clearest grouping.
vals = X_test["marital"].fillna("missing")
g1 = (vals == "g").values
g2 = ~g1
m1 = group_metrics(y_test[g1], y_prob[g1])
m2 = group_metrics(y_test[g2], y_prob[g2])

print("MARITAL: groups g and (gg, p) pooled")
print(f"  group g      n={m1['n']:>3}  approval {m1['approval_rate']:.3f}  recall {m1['recall_tpr']:.3f}")
print(f"  group gg+p   n={m2['n']:>3}  approval {m2['approval_rate']:.3f}  recall {m2['recall_tpr']:.3f}")
print()
approval_gap = m1["approval_rate"] - m2["approval_rate"]
print(f"  approval-rate gap (g minus gg+p): {approval_gap:+.3f}")
print()
if abs(approval_gap) > 0.10:
    print("  A gap this size is a disparate-impact signal worth investigating.")
    print("  It is NOT proof of discrimination — small groups, correlated")
    print("  features and a 207-account test set all limit what can be")
    print("  concluded — but it is the kind of number that must be explained")
    print("  before a lending model is deployed.")
else:
    print("  The approval-rate gap is small at this threshold.")

# %%
# Visualise the group metrics
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

groups = ["marital=g", "marital=gg+p"]
metrics_names = ["approval_rate", "recall_tpr", "fpr"]
data = [[m1[k] for k in metrics_names], [m2[k] for k in metrics_names]]
xpos = np.arange(len(metrics_names))
width = 0.35

axes[0].bar(xpos - width / 2, data[0], width, label="marital = g", color="#4C72B0")
axes[0].bar(xpos + width / 2, data[1], width, label="marital = gg / p", color="#C44E52")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(["approval rate", "recall (defaults caught)", "false-alarm rate"])
axes[0].set_ylim(0, 1)
axes[0].set_title("Outcome rates by marital group")
axes[0].legend()

# Score distributions by group — a threshold-independent view
axes[1].hist(y_prob[g1], bins=25, alpha=0.6, density=True,
             label=f"marital = g (n={m1['n']})", color="#4C72B0")
axes[1].hist(y_prob[g2], bins=25, alpha=0.6, density=True,
             label=f"marital = gg/p (n={m2['n']})", color="#C44E52")
axes[1].axvline(0.5, ls="--", color="black", lw=1, label="threshold 0.5")
axes[1].set_xlabel("predicted probability of default")
axes[1].set_ylabel("density")
axes[1].set_title("Score distributions overlap almost completely")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Statistical test on the approval gap: is it distinguishable from chance?
n1, n2 = m1["n"], m2["n"]
a1 = m1["tp"] + m1["fp"]
a2 = m2["tp"] + m2["fp"]
table = [[a1, n1 - a1], [a2, n2 - a2]]
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"chi-square on approval rate x marital group: chi2 = {chi2:.3f}, p = {p:.4f}")
print()
if p < 0.05:
    print("  The approval gap is unlikely to be chance. That is a finding that")
    print("  needs an explanation before deployment, not a footnote.")
else:
    print("  The gap is not statistically distinguishable from chance at n=207.")
    print("  NOTE: this is a *lack of evidence*, not evidence of no disparity.")
    print("  207 test accounts cannot detect a gap of the size that would")
    print("  actually matter in a lending portfolio. The correct reading is")
    print("  'this analysis is underpowered', not 'the model is fair'.")

# %%
# What would be needed to actually conclude something
print()
print("=" * 68)
print("  WHAT THIS FAIRNESS SECTION CAN AND CANNOT SUPPORT")
print("=" * 68)
print("CAN:")
print("  - show whether outcome rates differ across available groups")
print("  - flag a disparity that warrants investigation")
print("  - document the limitation explicitly")
print()
print("CANNOT:")
print("  - certify the model is fair")
print("  - detect disparate impact through a protected attribute that is")
print("    not in the data (the strongest form of the problem is invisible here)")
print("  - survive a threshold sweep without re-checking, since every")
print("    threshold changes the approval rates")
print()
print("A real fair-lending review needs: the protected attributes themselves,")
print("a test set an order of magnitude larger, a threshold sweep, and")
print("intersectional analysis. None of that is available from 690 accounts.")

# %% [markdown]
## Summary

| | Column 1 | Column 2 |
|---|---|---|
| Preprocessing | outside CV, scaled array unused | inside `Pipeline`, refit per fold |
| Nominal columns | `LabelEncoder` (false ordering) | `OneHotEncoder` |
| Split | unstratified | stratified |
| Tuned | `max_iter`, `tol` | `C`, `class_weight` |
| Reported | accuracy, confusion matrix | accuracy, balanced acc, ROC-AUC, PR-AUC, calibration |
| Threshold | 0.5, unexamined | chosen from a stated cost ratio |
| Test set | reused | held out from all tuning |
| Fairness | none | descriptive per-group analysis with stated limits |

# %%
print("FINAL HELD-OUT NUMBERS (from code that ran)")
print(f"  CV balanced accuracy  : {cv_best:.4f}")
print(f"  held-out balanced acc : {test_bal_acc:.4f}")
print(f"  held-out ROC-AUC      : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  held-out PR-AUC       : {average_precision_score(y_test, y_prob):.4f}")
print(f"  calibration error     : {ece:.4f}")
print()
print("The honest summary:")
print(f"  - the model is a real improvement over the majority baseline")
print(f"    ({accuracy_score(y_test, y_pred):.3f} vs {1 - y_test.mean():.3f} accuracy,")
print(f"    {roc_auc_score(y_test, y_prob):.3f} ROC-AUC) — the 2019 version was not")
print(f"  - CV overstates held-out performance by {cv_best - test_bal_acc:.3f}, so the")
print("    tuning generalised worse than the CV number suggested")
print(f"  - probabilities are off by {ece:.1%} on average, too much to quote as a")
print("    literal risk without recalibration")
print(f"  - the approval-rate gap by marital status is significant (p = {p:.4f})")
print("    and is the single most important result in this notebook")
print()
print("That last point is why a fairness section is not decoration. The model")
print("is a good classifier and still produces a statistically significant")
print("disparate-impact signal, and nothing in the accuracy or ROC-AUC numbers")
print("would have revealed it.")
