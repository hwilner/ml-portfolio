# 1 — Regression Discontinuity: Bank Debt Recovery (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** This reproduces the original 2019 analysis as
it was written: a scatter plot, two balance tests, and an OLS fit with a
threshold indicator.

It is kept because it is the honest starting point. The reasoning order here is
better than most working portfolios manage — *balance checks first, effect
estimate second* — and the mistakes in the next two notebooks are only
visible once you have seen what the naive version does.

Read this, then `02-col-2-2019-judgement.ipynb`, then
`03-col-3-2026-tools.ipynb`.

## The question

A bank assigns each debtor one of five recovery strategies based on the size of
the expected recovery. Accounts at or above **$1,000** receive a different,
more intensive strategy.

Does crossing that threshold change how much money actually comes back?

This is a **causal** question, not a predictive one. Debtors above and below the
threshold differ in ways that have nothing to do with the policy — they are
bigger accounts. So comparing group means directly is not an experiment. What
makes regression discontinuity work is that the comparison happens *locally*,
at the boundary, where accounts above and below are nearly indistinguishable.

## The data

1,882 accounts. `expected_recovery_amount` is the running variable that
determines assignment; `actual_recovery_amount` is the outcome.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA = Path("data/statistics/bank_data.csv")
THRESHOLD = 1000

df = pd.read_csv(DATA)
print(f"{len(df):,} accounts, {df.shape[1]} columns")
df.head()

# %%
df.dtypes

# %% [markdown]
## Step 1 — Look at the data before modelling it

The threshold creates two groups. Note how far apart the raw means are, and
keep that number in mind: it is the number a naive comparison would report, and
it is not a causal effect.

# %%
df["above_threshold"] = (df.expected_recovery_amount >= THRESHOLD).astype(int)

summary = df.groupby("above_threshold").agg(
    n=("actual_recovery_amount", "size"),
    expected_mean=("expected_recovery_amount", "mean"),
    actual_mean=("actual_recovery_amount", "mean"),
    actual_std=("actual_recovery_amount", "std"),
)
summary

# %% [markdown]
The naive comparison: mean actual recovery is **$518** below the threshold and
**$4,525** above it. An $4,006 "effect".

That number is almost entirely an artefact of account size. A $9,000 account
recovers more money than a $700 account whether or not the bank applies a
different strategy to it. This is why the design compares *near* the threshold
rather than everywhere.

A scale-free look at the same thing — recovery as a fraction of expected — is
more honest already:

# %%
df["recovery_ratio"] = df.actual_recovery_amount / df.expected_recovery_amount
df.groupby("above_threshold").recovery_ratio.agg(["size", "mean", "std"])

# %% [markdown]
## Step 2 — Plot around the threshold

The design assumption is **continuity**: if an account had been $5 below the
threshold rather than $5 above, it would have received the same strategy and
recovered a similar amount. A visible jump is the effect; a smooth curve
through the cut-off is evidence the design holds.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Full range — the scale difference dominates
ax = axes[0]
ax.scatter(
    df.expected_recovery_amount, df.actual_recovery_amount,
    s=8, alpha=0.3, color="#4C72B0", edgecolors="none",
)
ax.axvline(THRESHOLD, color="crimson", ls="--", lw=1.5, label="threshold = $1,000")
ax.set_xlabel("expected recovery ($)")
ax.set_ylabel("actual recovery ($)")
ax.set_title("Full range: the scale difference dominates")
ax.legend()

# Zoomed band — this is the window the design actually uses
ax = axes[1]
band = df[(df.expected_recovery_amount > 400) & (df.expected_recovery_amount < 1600)]
ax.scatter(
    band.expected_recovery_amount, band.actual_recovery_amount,
    s=14, alpha=0.55, color="#4C72B0", edgecolors="none",
)
ax.axvline(THRESHOLD, color="crimson", ls="--", lw=1.5)
ax.set_xlabel("expected recovery ($)")
ax.set_ylabel("actual recovery ($)")
ax.set_title("Band around the threshold")
plt.tight_layout()
plt.show()

# %%
# How much data is actually near the cut-off? This matters for everything below.
for lo, hi in [(0, 2000), (500, 1500), (800, 1200), (900, 1100)]:
    n = df.expected_recovery_amount.between(lo, hi).sum()
    print(f"  band [{lo:>4}, {hi:>4}]: {n:>5} accounts")

# %% [markdown]
## Step 3 — Balance checks: is anything else different at the threshold?

Before trusting a causal estimate, check whether accounts just below and just
above the threshold are actually comparable. If they differ systematically on
age, sex or strategy, the continuity assumption is in trouble.

# %%
# Age by recovery strategy — Kruskal-Wallis
groups = [
    g.age.values
    for _, g in df.groupby("recovery_strategy")
    if len(g) > 1
]
h, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis on age across {len(groups)} recovery strategies")
print(f"  H = {h:.3f},  p = {p:.3e}")
print("  -> age differs strongly by strategy" if p < 0.05 else "  -> no difference")

# %%
# Sex by recovery strategy — chi-square
sex_table = pd.crosstab(df.recovery_strategy, df.sex)
chi2, p, dof, expected = stats.chi2_contingency(sex_table)
print(f"Chi-square on sex x recovery strategy")
print(f"  chi2 = {chi2:.3f},  dof = {dof},  p = {p:.3f}")
print("  -> sex is balanced across strategies" if p > 0.05 else "  -> sex differs")
sex_table

# %% [markdown]
Age is strongly determined by recovery strategy. That is fine — the threshold
assigns *strategy*, so age differing by strategy is a consequence of the policy,
not a confound. The relevant check is whether **age is continuous through the
threshold**, which is what the plot in Step 4 shows.

# %%
# Continuity of the running variable: age on each side, within a narrow band
band = df[df.expected_recovery_amount.between(700, 1300)]
low_age = band.loc[band.expected_recovery_amount < THRESHOLD, "age"]
high_age = band.loc[band.expected_recovery_amount >= THRESHOLD, "age"]
t, p = stats.ttest_ind(low_age, high_age, equal_var=False)
print(f"Age in band [700, 1300]: below mean {low_age.mean():.2f} (n={len(low_age)}), "
      f"above {high_age.mean():.2f} (n={len(high_age)})")
print(f"  Welch t-test: t = {t:.3f}, p = {p:.3f}  -> "
      + ("continuous" if p > 0.05 else "NOT continuous at the threshold"))

# %% [markdown]
## Step 4 — The estimate: OLS with a threshold indicator

The original model: actual recovery regressed on a `post` indicator (1 if at or
above $1,000) plus the running variable. The `post` coefficient is the estimated
effect of crossing the threshold.

# %%
m_ols = smf.ols("actual_recovery_amount ~ above_threshold + expected_recovery_amount",
                data=df).fit()
print(m_ols.summary2().tables[1])

# %% [markdown]
The coefficient on `above_threshold` comes out at **−$641**: the model says
accounts just above $1,000 recover *less* than accounts just below it.

**That is the wrong sign.** Within any narrow band around the cut-off, the raw
means run the other way — at ±$200, accounts below the threshold recover $575
and accounts above recover $978, a **positive** jump of about $403.

The discrepancy is the whole lesson of this notebook, and the original 2019
analysis never noticed it, because it reported the coefficient and moved on
without ever comparing it to the local descriptive statistic it had already
computed. Three problems compound:

1. **A single slope fitted over the entire range** (expected recovery spans
   $5 to $10,000) is dominated by large accounts and extrapolates badly to the
   boundary. The fitted slope is $2.09 per dollar above the cut-off but $0.67
   below it — a 3x difference that a single slope cannot represent.
2. **No bandwidth restriction.** A regression-discontinuity estimate is only
   valid *locally*; fitting a line to the whole distribution asks it to
   extrapolate from $10,000 accounts to a $1,000 boundary.
3. **Homoskedastic standard errors** on a strongly right-skewed outcome.

None of these raise an error. The model returns a highly significant result,
with p = 1.3e-17, and it has the wrong sign.

# %%
# Robust standard errors — note these are *smaller*, which is the opposite of
# what a textbook intuition about skewness predicts. Robustness to the variance
# assumption and correctness of the functional form are separate problems, and
# fixing the first does not touch the second.
m_ols_robust = smf.ols(
    "actual_recovery_amount ~ above_threshold + expected_recovery_amount", data=df
).fit(cov_type="HC3")
ols_se = m_ols.bse["above_threshold"]
robust_se = m_ols_robust.bse["above_threshold"]
print(f"Standard OLS SE : {ols_se:.4f}")
print(f"HC3 robust SE   : {robust_se:.4f}")
print(f"Ratio           : {robust_se / ols_se:.2f}x  (robust is SMALLER here)")
print()
print("The p-value was already tiny, so this changes nothing about the")
print("conclusion — which is still the wrong sign.")
print(m_ols_robust.summary2().tables[1])

# %%
# Heteroskedasticity is real — the spread grows with the running variable
df["abs_resid"] = m_ols.resid.abs()
print("Mean |residual| by expected-recovery quintile:")
print(df.groupby(pd.qcut(df.expected_recovery_amount, 5, duplicates="drop"),
                 observed=True).abs_resid.mean())
print()
print("The bottom quintile (small accounts) has far larger absolute errors")
print("than the top — the variance assumption is violated, as expected.")

# %%
# The decisive check the original analysis never ran: compare the regression
# coefficient against the local descriptive statistic, in the same place.
print("=" * 62)
print("  REGRESSION COEFFICIENT vs LOCAL MEANS")
print("=" * 62)
for b in [200, 500, 1000]:
    s = df[df.expected_recovery_amount.between(THRESHOLD - b, THRESHOLD + b)]
    below = s.loc[s.above_threshold == 0, "actual_recovery_amount"]
    above = s.loc[s.above_threshold == 1, "actual_recovery_amount"]
    print(f"  band +/-${b:>4}: raw jump = ${above.mean() - below.mean():>8.2f}  "
          f"(n={len(s)})")
print(f"  full-sample OLS coefficient        = ${m_ols.params['above_threshold']:>8.2f}")
print("=" * 62)
print()
print("Every local window gives a POSITIVE jump. The full-sample regression")
print("gives a NEGATIVE one. One of them is wrong, and the sign disagreement")
print("is the diagnostic that something is badly misspecified.")

# %% [markdown]
## Step 5 — The original robustness check: a narrower band

The 2019 notebook re-ran the model on a restricted window. Restricting the
bandwidth is the standard first sensitivity check.

# %%
for lo, hi in [(0, 2000), (500, 1500), (800, 1200), (900, 1100)]:
    sub = df[df.expected_recovery_amount.between(lo, hi)]
    m = smf.ols("actual_recovery_amount ~ above_threshold + expected_recovery_amount",
                data=sub).fit(cov_type="HC3")
    coef = m.params["above_threshold"]
    ci = m.conf_int().loc["above_threshold"]
    print(f"  band [{lo:>4}, {hi:>4}]  n={len(sub):>5}  "
          f"effect = ${coef:>8.2f}   95% CI [{ci[0]:.1f}, {ci[1]:.1f}]  p = {m.pvalues['above_threshold']:.4f}")

# %% [markdown]
## What this notebook concluded, and what it missed

**Concluded:** crossing the $1,000 threshold *reduces* actual recovery by about
$641, with p = 1.3e-17, "robust to a narrower bandwidth".

**That conclusion is wrong**, and the next notebook is the correction. The
single most damning detail is the one the original analysis had in its hands
and did not look at: the local means say the opposite sign.

**Missed — the four checks that decide whether that is causal:**

| Missing check | Why it matters |
|---|---|
| **Compare the estimate to the local means** | The coefficient is −$641; every local window says **+$403**. A sign reversal is a specification failure, and it is visible in two lines of code |
| Bandwidth restriction | An RD estimate is only valid locally. Fitting over a $5–$10,000 range extrapolates to a $1,000 boundary |
| Separate slopes either side of the cut-off | The fitted slope is $0.67 below the cut-off and $2.09 above — a 3x difference one line cannot express |
| Manipulation test (McCrary density) | Debtors may **choose** to stay under $1,000. If they do, the comparison is confounded |
| Placebo cut-offs at $1,500 / $2,000 | If an "effect" appears where no policy changes, the estimator is finding something else |

None of these needed a library newer than 2019. All of them are in the next
notebook.

# %%
# Two-bandwidth checking is *not* bandwidth selection. A stated rule is.
# What the 2019 version did: try two windows, keep the one that looked good.
# What should happen: choose the bandwidth by a rule fixed in advance, and
# report the whole selection curve. That is the difference between a
# sensitivity check and an estimate.
print("The 2019 approach used 2 hand-picked windows.")
print("The next notebook selects bandwidth by cross-validation over a grid.")
