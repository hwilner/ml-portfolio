# Statistics & causal inference

## Regression discontinuity: bank debt recovery

**File:** [`01-regression-discontinuity-bank-recovery.ipynb`](01-regression-discontinuity-bank-recovery.ipynb)
**Status:** ✅ runs end to end
**Competencies:** causal inference ●●● · hypothesis testing ●●● · regression ●●●

### The question

A bank escalates its collection strategy once a debt crosses a threshold
($1,000, $2,000, $3,000, $5,000). Does that policy *change how much the bank
actually recovers*?

This is not a prediction problem. The goal is a **causal** estimate of the
policy's effect, which means dealing with the fact that accounts above a
threshold differ systematically from accounts below it — bigger debts are sent
to more aggressive collectors. Simple comparison of the two groups is
confounded.

### Why a regression discontinuity design applies

Right at a threshold, accounts above and below are *nearly* identical: same
product, same customer base, same era. The only thing that differs is which
side of the line they fell on. That gives a quasi-experimental comparison
without needing randomisation — provided nothing else changes discontinuously
at the threshold. Checking that is the first thing the notebook does.

### What the notebook does

1. **Graphical EDA** — scatter plots around the $1,000 threshold to see whether
   a visible discontinuity exists at all.
2. **Balance tests.** The design is only credible if covariates are *not*
   themselves discontinuous at the threshold:
   - Kruskal–Wallis on age across recovery strategies (non-parametric, since
     the groups are small and skewed).
   - Chi-square on the sex distribution across strategies.
   Both fail to reject, supporting the design's assumptions.
3. **Effect estimation with OLS** via `statsmodels`, adding an indicator for
   `expected_recovery_amount >= 1000`. The indicator's coefficient is the
   estimated jump in recovery at the threshold.
4. **Bandwidth sensitivity** — re-estimates on a narrower $950–1050 window. If
   the effect holds, it is not an artefact of the wide window.

### What it demonstrates

- Knowing when a question is causal rather than predictive, and choosing a
  design that matches.
- Testing assumptions *before* trusting an estimate, not after.
- Knowing which test fits which data: Kruskal–Wallis over t-test for small,
  skewed groups; chi-square for categorical balance.
- Reporting sensitivity to a tuning parameter.

### Reproducing

```bash
pip install numpy pandas scipy statsmodels matplotlib
jupyter notebook 01-regression-discontinuity-bank-recovery.ipynb
```

Dataset is committed at [`data/statistics/bank_data.csv`](../../data/statistics/bank_data.csv).
