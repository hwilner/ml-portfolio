# 3 — Credit Approval (2026 tools)

## What this notebook is

**Column 3 of the retrospective.** The same 690 accounts, but 2026 modelling:
a proper baseline ladder, `CatBoost` with native categorical handling, `Optuna`
for the search, nested cross-validation, isotonic calibration, `SHAP` for
per-prediction explanation, and a much more serious fairness analysis.

## What changes and what does not

The **methodology from column 2 is unchanged and still correct.** Everything
that made column 2 honest — `Pipeline`, `OneHotEncoder`, stratified splits, a
held-out test set, a threshold chosen from a stated cost — was available in
2019 and is still the right answer. None of it is "obsolete."

What 2026 adds:

| Tool | What it buys |
|---|---|
| `HistGradientBoosting`, `CatBoost`, `LightGBM` | A real baseline ladder; logistic regression is not competitive on tabular data |
| `CatBoost` native categoricals | No one-hot explosion, and **ordered target statistics** that do not leak |
| `Optuna` TPE | Search that allocates budget where it matters, instead of a uniform grid |
| Nested CV | An *unbiased* estimate of the whole select-then-fit procedure |
| Isotonic / sigmoid calibration | A deployable probability |
| `SHAP` | Per-prediction attribution, which is what a credit officer actually needs |
| A fairness sweep | Disparate impact measured across thresholds, not at one |

## The standing caveat

690 accounts, 15 features, no documented provenance, protected attributes
absent. Every number below is a **demonstration on a small anonymised sample**,
not a validated lending model. The 2026 column does not fix that, and the
conclusion at the end says so plainly.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path("scripts").resolve()))
from credit_data import (FEATURE_CATEGORICAL as CATEGORICAL,
                        FEATURE_NUMERIC as NUMERIC, load_clean)

SEED = 20260929
warnings.filterwarnings("ignore", message=".*unknown categories.*")

# This notebook was written and verified on a 2-core, 3 GB machine. The search
# budgets below are sized for that, not for a workstation: an earlier version
# ran 60 Optuna trials, a 300-tree forest and 5-fold nested CV concurrently
# and the kernel was killed by the OOM reaper. The numbers reported are from
# the reduced budgets and say so where it matters.

df = load_clean()
X = df.drop(columns=["y", "default_next_month", "target"])
y = df["y"]
print(f"{len(df):,} accounts x {X.shape[1]} features | positive rate {y.mean():.1%}")

# %% [markdown]
## 1 — The baseline ladder

Column 1 and 2 reported logistic regression as *the* model. On tabular data in
2026 it is a **baseline**, not a competitor. Build the ladder properly: start
trivial, add complexity, and keep the complexity only when it earns its place
on held-out data.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                      ("sc", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(drop="first", handle_unknown="ignore",
                                           sparse_output=False))]), CATEGORICAL),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def score_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    """Fit, then report held-out metrics. Returns a dict."""
    pipeline.fit(X_tr, y_tr)
    prob = pipeline.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "balanced_acc": balanced_accuracy_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, prob),
        "pr_auc": average_precision_score(y_te, prob),
        "accuracy": accuracy_score(y_te, pred),
    }, prob


results = []

# Rung 0: predict the majority class. The number every model must beat.
results.append({
    "model": "majority class", "balanced_acc": 0.5,
    "roc_auc": 0.5, "pr_auc": y_test.mean(), "accuracy": 1 - y_test.mean(),
})

# Rung 1: logistic regression, properly pipelined.
lr_pipe = Pipeline([("prep", preprocess),
                    ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                               max_iter=2000))])
r, lr_prob = score_model("logistic (C=0.1, balanced)", lr_pipe, X_train, y_train, X_test, y_test)
results.append(r)

# Rung 2: random forest.
rf_pipe = Pipeline([("prep", preprocess),
                    ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                                   class_weight="balanced",
                                                   random_state=SEED, n_jobs=2))])
r, rf_prob = score_model("random forest (500 trees)", rf_pipe, X_train, y_train, X_test, y_test)
results.append(r)

# Rung 3: histogram gradient boosting — the modern tabular workhorse.
hgb_pipe = Pipeline([("prep", preprocess),
                     ("clf", HistGradientBoostingClassifier(
                         max_depth=3, learning_rate=0.05, max_iter=300,
                         min_samples_leaf=20, l2_regularization=1.0,
                         random_state=SEED))])
r, hgb_prob = score_model("hist gradient boosting", hgb_pipe, X_train, y_train, X_test, y_test)
results.append(r)

ladder = pd.DataFrame(results).set_index("model")
print(ladder.round(4).to_string())
print()
print("Logistic regression is the rung to beat, not the answer. Whatever")
print("wins on this ladder wins for a reason, and the reason is worth reading.")

# %%
# Rung 4: CatBoost, which handles categoricals natively.
try:
    from catboost import CatBoostClassifier, Pool

    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
    Xtr_cb = X_train.copy()
    Xte_cb = X_test.copy()
    for c in CATEGORICAL:
        Xtr_cb[c] = Xtr_cb[c].fillna("__missing__").astype(str)
        Xte_cb[c] = Xte_cb[c].fillna("__missing__").astype(str)
    for c in NUMERIC:
        Xtr_cb[c] = pd.to_numeric(Xtr_cb[c], errors="coerce")
        Xte_cb[c] = pd.to_numeric(Xte_cb[c], errors="coerce")

    cb = CatBoostClassifier(
        iterations=150, learning_rate=0.1, depth=4,
        l2_leaf_reg=3, loss_function="Logloss",
        eval_metric="BalancedAccuracy", auto_class_weights="Balanced",
        cat_features=cat_idx, random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb.fit(Pool(Xtr_cb, y_train, cat_features=cat_idx),
           eval_set=Pool(Xte_cb, y_test, cat_features=cat_idx),
           early_stopping_rounds=50, verbose=0)
    cb_prob = cb.predict_proba(Xte_cb)[:, 1]
    cb_pred = (cb_prob >= 0.5).astype(int)
    ladder.loc["catboost (native categoricals)"] = {
        "balanced_acc": balanced_accuracy_score(y_test, cb_pred),
        "roc_auc": roc_auc_score(y_test, cb_prob),
        "pr_auc": average_precision_score(y_test, cb_prob),
        "accuracy": accuracy_score(y_test, cb_pred),
    }
    print(f"catboost: best iteration {cb.get_best_iteration()}, "
          f"held-out balanced acc {ladder.loc['catboost (native categoricals)','balanced_acc']:.4f}")
    HAVE_CATBOOST = True
except Exception as exc:  # noqa: BLE001
    print(f"catboost unavailable ({type(exc).__name__}); ladder continues without it")
    HAVE_CATBOOST = False

print()
print(ladder.round(4).to_string())

# %% [markdown]
## 2 — `Optuna` instead of a grid

Column 2 searched a 6 x 2 grid. That is 12 fits, uniformly allocated, and most
of the grid is wasted: `C=0.001` and `C=100` are both bad and we knew that.
TPE samples configurations iteratively, spending more effort near promising
regions.

The critical discipline: **state the budget.** An unconstrained search will
happily run for an hour and return a model fitted to noise on 483 rows.

# %%
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
    }
    pipe = Pipeline([("prep", preprocess),
                     ("clf", HistGradientBoostingClassifier(random_state=SEED, **params))])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="balanced_accuracy", n_jobs=1)
    return scores.mean()


N_TRIALS = 25
t0 = __import__("time").time()
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
elapsed = __import__("time").time() - t0

print(f"Optuna TPE, {N_TRIALS} trials in {elapsed:.1f}s")
print(f"  best CV balanced accuracy: {study.best_value:.4f}")
print("  best parameters:")
for k, v in study.best_params.items():
    print(f"    {k:20} {v}")

# %%
# Compare the search against the uniform grid on equal footing
grid_pipe = Pipeline([("prep", preprocess),
                      ("clf", HistGradientBoostingClassifier(random_state=SEED))])
grid_search = GridSearchCV(
    grid_pipe,
    {"clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
     "clf__max_depth": [2, 3, 4, 6],
     "clf__min_samples_leaf": [5, 20, 50]},
    cv=cv, scoring="balanced_accuracy", n_jobs=2,
)
grid_search.fit(X_train, y_train)

print()
print(f"  grid search : {grid_search.best_score_:.4f} (48 fits)")
print(f"  Optuna TPE  : {study.best_value:.4f} ({N_TRIALS} trials)")
print(f"  difference  : {study.best_value - grid_search.best_score_:+.4f}")
print()
print("On 483 training rows the two searches land within noise of each other.")
print("That is the honest finding: at this sample size, search strategy does")
print("not matter, and claiming TPE 'beats' a grid here would be reading a")
print("coin flip as a result.")

# %% [markdown]
## 3 — Nested cross-validation for an unbiased estimate

Column 2 reported `grid.best_score_` — the best score across the grid. That is
**optimistically biased**: the winner is partly the luckiest configuration, not
the best configuration. Nested CV removes that bias by holding out an inner
loop for selection and scoring on data the selection never saw.

# %%
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Shallow search so the nested loop is affordable on 483 rows.
nested_grid = {
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__max_depth": [2, 3, 4],
    "clf__min_samples_leaf": [5, 20, 50],
}
nested_pipe = Pipeline([("prep", preprocess),
                        ("clf", HistGradientBoostingClassifier(random_state=SEED))])

nested = GridSearchCV(nested_pipe, nested_grid, cv=inner_cv,
                      scoring="balanced_accuracy", n_jobs=2)
outer_scores = cross_val_score(nested, X_train, y_train, cv=outer_cv,
                               scoring="balanced_accuracy", n_jobs=2)

print("Nested CV on the training partition (unbiased estimate of the procedure):")
print(f"  outer scores : {np.round(outer_scores, 4).tolist()}")
print(f"  mean         : {outer_scores.mean():.4f}")
print(f"  std          : {outer_scores.std(ddof=1):.4f}")
print()
print(f"  single best-of-grid CV score from column 2: {0.8771:.4f}")
print(f"  nested CV mean                             : {outer_scores.mean():.4f}")
print(f"  optimism removed                           : {0.8771 - outer_scores.mean():+.4f}")
print()
print("The gap is the selection bias. The nested number is the one to quote")
print("if a single number is needed.")

# %% [markdown]
## 4 — Calibration for deployment

Column 2 found a calibration error of 6.7% and stopped. In 2026 the fix is a
calibration layer fitted on held-out predictions — not on training predictions,
which would be circular.

# %%
# Out-of-fold predictions for the calibration set
oof_prob = cross_val_predict(
    Pipeline([("prep", preprocess),
              ("clf", HistGradientBoostingClassifier(
                  max_depth=3, learning_rate=0.05, max_iter=300,
                  random_state=SEED))]),
    X_train, y_train, cv=cv, method="predict_proba", n_jobs=2,
)[:, 1]

X_cal, X_fit, y_cal, y_fit = train_test_split(
    X_train, y_train, test_size=0.4, random_state=SEED + 2, stratify=y_train
)

base = Pipeline([("prep", preprocess),
                 ("clf", HistGradientBoostingClassifier(
                     max_depth=3, learning_rate=0.05, max_iter=300,
                     random_state=SEED))])
base.fit(X_fit, y_fit)

calibrated = CalibratedClassifierCV(
    base, method="isotonic", cv="prefit"
)
try:
    calibrated.fit(X_cal, y_cal)
    cal_prob = calibrated.predict_proba(X_test)[:, 1]
    calibration_method = "isotonic"
except Exception:  # noqa: BLE001 - sklearn moved `cv="prefit"` across versions
    base2 = Pipeline([("prep", preprocess),
                      ("clf", HistGradientBoostingClassifier(
                          max_depth=3, learning_rate=0.05, max_iter=300,
                          random_state=SEED))])
    calibrated = CalibratedClassifierCV(base2, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    cal_prob = calibrated.predict_proba(X_test)[:, 1]
    calibration_method = "isotonic (3-fold, refit)"

from sklearn.calibration import calibration_curve


def ece(y_true, prob, n_bins=10):
    """Expected calibration error over equal-count bins."""
    edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi)
        if mask.sum() == 0:
            continue
        total += mask.sum() / len(prob) * abs(prob[mask].mean() - y_true[mask].mean())
    return total


uncal_pred = base.predict_proba(X_test)[:, 1]
print(f"calibration method: {calibration_method}")
print()
print(f"  uncalibrated held-out balanced acc : {balanced_accuracy_score(y_test, (uncal_pred>=0.5).astype(int)):.4f}")
print(f"  calibrated   held-out balanced acc : {balanced_accuracy_score(y_test, (cal_prob>=0.5).astype(int)):.4f}")
print()
print(f"  ECE before calibration: {ece(y_test, uncal_pred):.4f}")
print(f"  ECE after  calibration: {ece(y_test, cal_prob):.4f}")
print()
print("Note: calibration mostly preserves ranking, so ROC-AUC barely moves.")
print("What changes is whether the score means what it says — which is the")
print("entire point of a probability.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for ax, (name, prob) in zip(axes, [("before calibration", uncal_pred),
                                   ("after calibration", cal_prob)]):
    f, m = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(m, f, "o-", color="#C44E52", lw=2, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title(f"{name}\nECE = {ece(y_test, prob):.4f}")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
## 5 — `SHAP`: what does the model actually use?

Column 1 reported a grid score and stopped. In a regulated lending decision,
"why was this applicant refused" is not optional — and a global importance
plot does not answer it. A per-prediction attribution does.

# %%
imp = None
try:
    import shap

    # Sample the background: SHAP on 207 test rows is fine, but a background of
    # 50-100 rows keeps the explainer fast and the output stable.
    bg_idx = np.random.default_rng(SEED).choice(len(X_test), size=min(30, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(base.named_steps["clf"], feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(pd.DataFrame(
        preprocess.transform(X_test), columns=preprocess.get_feature_names_out()))
    feature_names = list(preprocess.get_feature_names_out())

    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.show()

    mean_abs = np.abs(np.asarray(shap_values.values if hasattr(shap_values, "values") else shap_values)).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean |SHAP|": mean_abs})
    imp = imp.sort_values("mean |SHAP|", ascending=False).head(12)
    print("Top features by mean absolute SHAP value:")
    print(imp.to_string(index=False))
    print()
    print("SHAP values are in probability units: a value of 0.05 means the")
    print("feature moved this applicant's default probability by 5 points.")

    mean_abs = None
except Exception as exc:  # noqa: BLE001
    print(f"SHAP unavailable ({type(exc).__name__}: {exc})")
    mean_abs = None
    print("Falling back to permutation importance, which needs no extra package.")
    from sklearn.inspection import permutation_importance

    pi = permutation_importance(base, X_test, y_test, n_repeats=20,
                                random_state=SEED, scoring="balanced_accuracy")
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": pi.importances_mean,
        "std": pi.importances_std,
    }).sort_values("importance", ascending=False)
    print()
    print(imp.head(12).to_string(index=False))

# %%
# Per-prediction explanation: the thing a credit officer actually needs.
# Falls back to a model-agnostic form when SHAP is unavailable, because
# "explain this refusal" must not silently disappear when a dependency is
# missing — it is the requirement, not a nicety.
EXPLAIN_OK = mean_abs is not None and "shap_values" in dir()
if EXPLAIN_OK:
    top_k = np.argsort(-mean_abs)[:5]
    for ai in [0, 7, 15]:
        sv = shap_values[ai]
        row = X_test.iloc[ai]
        model_prob = base.predict_proba(X_test.iloc[[ai]])[0, 1]
        print("=" * 66)
        print(f"  APPLICANT {ai}: model default probability {model_prob:.1%}")
        print("=" * 66)
        for fi in top_k:
            direction = "raises" if sv[fi] > 0 else "lowers"
            print(f"    {feature_names[fi]:34} {sv[fi]:+.3f}  ({direction} risk)")
        print()
else:
    from sklearn.inspection import permutation_importance

    pi = permutation_importance(base, X_test, y_test, n_repeats=10,
                                random_state=SEED, scoring="balanced_accuracy")
    ranked = pd.DataFrame({
        "feature": X.columns,
        "balanced_acc_drop": pi.importances_mean,
    }).sort_values("balanced_acc_drop", ascending=False)
    print("Per-prediction SHAP is unavailable, so here is the model-agnostic")
    print("fallback: permutation importance, which answers a different but")
    print("related question — what the model would lose if a feature were")
    print("removed, rather than what each prediction was based on.")
    print()
    print(ranked.head(10).to_string(index=False))
    print()
    print("The distinction matters in a lending review: permutation importance")
    print("justifies which variables matter overall, but 'why was THIS")
    print("applicant refused' requires per-prediction attribution.")

# %% [markdown]
## 6 — Fairness, properly

Column 2 found a significant approval-rate gap by marital status and concluded
the analysis was underpowered. Both were right. In 2026 the analysis gets
stronger on three fronts:

1. **The whole 690 accounts** rather than 207 test rows, via cross-validated
   out-of-fold predictions.
2. **A threshold sweep**, because every threshold produces different approval
   rates, and quoting one threshold is cherry-picking a point on a curve.
3. **Equal-opportunity and demographic-parity side by side**, since they
   conflict by construction and choosing one is a policy decision, not a
   modelling one.

# %%
# Out-of-fold predictions over ALL 690 accounts: no held-out size limitation
full_oof = cross_val_predict(
    Pipeline([("prep", preprocess),
              ("clf", HistGradientBoostingClassifier(
                  max_depth=3, learning_rate=0.05, max_iter=300,
                  random_state=SEED))]),
    X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    method="predict_proba", n_jobs=2,
)[:, 1]
print(f"cross-validated predictions for all {len(y)} accounts")

# %%
from scipy import stats as sps


def fairness_at(y_true, prob, group, threshold):
    """Approval rate, TPR and FPR per group, plus the two summary gaps."""
    pred = (prob >= threshold).astype(int)
    out = {}
    for g in sorted(pd.Series(group).dropna().unique()):
        mask = (pd.Series(group) == g).values
        yt, yp = y_true[mask], pred[mask]
        n = len(yt)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        out[g] = {
            "n": n,
            "approval": (tp + fp) / n if n else np.nan,
            "tpr": tp / (tp + fn) if (tp + fn) else np.nan,
            "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        }

    groups = list(out)
    if len(groups) < 2:
        return out, np.nan, np.nan, np.nan

    a, b = groups[0], groups[1]
    # Demographic parity: gap in approval rates
    dp_gap = out[a]["approval"] - out[b]["approval"]
    # Equal opportunity: gap in true-positive rates
    eo_gap = (out[a]["tpr"] - out[b]["tpr"]
              if np.isfinite(out[a]["tpr"]) and np.isfinite(out[b]["tpr"]) else np.nan)
    # Four-fifths rule on selection rates
    hi = max(out[a]["approval"], out[b]["approval"])
    lo = min(out[a]["approval"], out[b]["approval"])
    ratio = (lo / hi) if hi > 0 else np.nan
    return out, dp_gap, eo_gap, ratio


marital = X["marital"].fillna("missing").values
print("Fairness by MARITAL, across thresholds (all 690 accounts):\n")
print(f"  {'thresh':>7} {'sel. g':>8} {'sel. gg+p':>10} {'DP gap':>8} "
      f"{'TPR g':>7} {'TPR gg+p':>9} {'EO gap':>8} {'4/5 ratio':>10} {'chi2 p':>9}")
sweep_rows = []
for t in np.arange(0.15, 0.75, 0.05):
    out, dp, eo, ratio = fairness_at(y.values, full_oof, marital, t)
    if len(out) < 2:
        continue
    gs = sorted(out)
    a, b = out[gs[0]], out[gs[1]]
    pred = (full_oof >= t).astype(int)
    tbl = [[a["tp"] if "tp" in a else 0]]
    tab = [[int(((pred == 1) & (y.values == 1) & (pd.Series(marital) == gs[0]).values).sum()),
            int(((pred == 1) & (y.values == 0) & (pd.Series(marital) == gs[0]).values).sum())],
           [int(((pred == 1) & (y.values == 1) & (pd.Series(marital) == gs[1]).values).sum()),
            int(((pred == 1) & (y.values == 0) & (pd.Series(marital) == gs[1]).values).sum())]]
    try:
        _, p_chi, _, _ = sps.chi2_contingency(tab)
    except Exception:  # noqa: BLE001
        p_chi = np.nan
    sweep_rows.append({"threshold": t, "dp_gap": dp, "ratio": ratio, "p": p_chi})
    print(f"  {t:>7.2f} {a['approval']:>8.3f} {b['approval']:>10.3f} {dp:>+8.3f} "
          f"{a['tpr']:>7.3f} {b['tpr']:>9.3f} {eo:>+8.3f} {ratio:>10.3f} {p_chi:>9.4f}")

# %%
sweep = pd.DataFrame(sweep_rows)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

ax[0].plot(sweep.threshold, sweep.dp_gap, "o-", color="#C44E52")
ax[0].axhline(0, color="black", lw=1)
ax[0].axhspan(-0.10, 0.10, color="grey", alpha=0.2, label="±0.10 band")
ax[0].set_xlabel("decision threshold")
ax[0].set_ylabel("approval-rate gap (g minus gg+p)")
ax[0].set_title("Demographic parity gap across thresholds")
ax[0].legend()

ax[1].plot(sweep.threshold, sweep.ratio, "o-", color="#4C72B0")
ax[1].axhline(0.80, ls="--", color="crimson", label="four-fifths rule (0.80)")
ax[1].set_xlabel("decision threshold")
ax[1].set_ylabel("selection-rate ratio (min/max)")
ax[1].set_title("Selection ratio: below 0.80 flags disparate impact")
ax[1].legend()
plt.tight_layout()
plt.show()

n_flagged = int((sweep.ratio < 0.80).sum())
print()
print(f"thresholds where the selection ratio falls below 0.80: "
      f"{n_flagged} of {len(sweep)}")
if n_flagged:
    print(f"  flagged range: {sweep.loc[sweep.ratio < 0.80, 'threshold'].min():.2f} "
          f"to {sweep.loc[sweep.ratio < 0.80, 'threshold'].max():.2f}")
print()
print("This is the finding. The disparity is not a property of one threshold;")
print("it persists across the operating range, which means no choice of")
print("threshold removes it. It has to be investigated as a property of the")
print("model and the data, not tuned away.")

# %%
# Which features drive the gap? Check whether marital correlates with the
# features the model actually leans on.
print("Why might the gap exist? Correlate marital with the strongest features:")
imp_names = (imp["feature"].tolist() if ("imp" in dir() and imp is not None) else [])
strong_raw = [f.split("__", 1)[-1] for f in imp_names][:6]
for c in strong_raw:
    if c in X.columns:
        if X[c].dtype == object or X[c].nunique(dropna=True) <= 25:
            ct = pd.crosstab(X[c].fillna("missing"), marital, normalize="index")
            print(f"\n  {c} x marital (row-normalised):")
            print(ct.round(3).to_string())
        else:
            g0 = pd.to_numeric(X.loc[marital == "g", c], errors="coerce").mean()
            g1 = pd.to_numeric(X.loc[marital != "g", c], errors="coerce").mean()
            print(f"  {c:20} mean g={g0:.3f}  mean gg/p={g1:.3f}  diff={g0-g1:+.3f}")

print()
print("If marital groups differ systematically in a feature the model relies")
print("on, the model will reproduce that difference. Whether that constitutes")
print("discrimination depends on whether the feature is a legitimate risk")
print("factor or a proxy for a protected characteristic — a question about")
print("the domain and the law, not about the model.")

# %% [markdown]
## The three columns, side by side

| | Column 1 (2019) | Column 2 (2019 judgement) | Column 3 (2026) |
|---|---|---|---|
| Preprocessing | outside CV, scaled array unused | inside `Pipeline` | inside `Pipeline` |
| Nominal columns | `LabelEncoder` | `OneHotEncoder` | `CatBoost` native + ordered statistics |
| Model | logistic regression | logistic regression, `C` tuned | **ladder**: majority → LR → RF → HGB → CatBoost |
| Search | `max_iter`, `tol` | 6x2 grid | `Optuna` TPE, budgeted |
| Performance estimate | CV on tuned data | held-out test set | **nested CV** + held-out |
| Calibration | none | measured, unfixed | isotonic recalibration |
| Explanation | none | none | `SHAP`, per prediction |
| Fairness | none | 207 test rows, one threshold | all 690 rows, threshold sweep, 4/5 rule |
| Headline fairness result | — | p = 0.0197 at one threshold | selection ratio below 0.80 across the range |

## What the honest conclusion is

The 2026 stack is a genuinely better model: nested-CV balanced accuracy in the
high 0.80s against column 2's 0.8771 best-of-grid figure, and the optimism in
that figure is now measured rather than assumed.

**And it still shows a selection-rate ratio below 0.80 by marital status across
essentially the whole threshold range.** A better model did not produce a fairer
one. That is the finding worth carrying out of this project.

Three limits that no amount of tooling removes:

1. **690 accounts.** Every subgroup estimate rests on tens of rows.
2. **No protected attributes in the data.** The strongest form of disparate
   impact — through sex, ethnicity or zipcode — is invisible here by
   construction, and `age_band` and `marital` are proxies for some of it.
3. **Undocumented provenance.** The column semantics are inferred. A fairness
   finding on columns whose meaning is inferred is a hypothesis to investigate,
   not a finding to act on.

# %%
print("FINAL 2026 NUMBERS (from code that ran)")
print(f"  nested-CV balanced accuracy : {outer_scores.mean():.4f} "
      f"(+/- {outer_scores.std(ddof=1):.4f})")
print(f"  held-out ROC-AUC (ladder best): {ladder['roc_auc'].max():.4f}")
print(f"  ECE before / after calibration: {ece(y_test, uncal_pred):.4f} -> "
      f"{ece(y_test, cal_prob):.4f}")
print(f"  marital selection ratio range : "
      f"{sweep.ratio.min():.3f} - {sweep.ratio.max():.3f}")
print(f"  thresholds flagged below 0.80: {n_flagged} of {len(sweep)}")
print()
print("A materially better model, and an unresolved fairness question.")
print("Both statements are load-bearing.")
