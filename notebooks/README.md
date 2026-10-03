# Notebooks

Numbered prefixes set the reading order.

**All 19 notebooks execute end to end.** Getting there meant fixing real bugs —
including a decision tree that could never use its first feature, and a
cross-validation run that compared a model against itself. The full record is
in [`BUGFIXES.md`](../BUGFIXES.md).

Where a dataset cannot be downloaded, the notebook generates a **clearly-labelled
synthetic stand-in**, prints a warning, and titles every figure `SYNTHETIC DATA`.
A perfect score on generated shapes is not evidence, and the notebooks say so
rather than letting the number stand.

| Folder | Area | Contents |
|---|---|---|
| [`01-nlp/`](01-nlp/) | NLP | LDA topic modelling, NYSIIS phonetic matching |
| [`02-statistics/`](02-statistics/) | Statistics | Regression discontinuity design |
| [`03-machine-learning/`](03-machine-learning/) | Classical ML | PCA + genre classification, credit approval, TPOT AutoML |
| [`04-computer-vision/`](04-computer-vision/) | Computer vision | Image manipulation, HOG + SVM, ASL CNN |
| [`05-algorithms-from-scratch/`](05-algorithms-from-scratch/) | Algorithms | Perceptron, gradient descent, momentum, decision tree, backprop |
| [`06-sql/`](06-sql/) | SQL | World Bank debt analysis |

## Which of these are worth your time

**Read these:**

- [`02-statistics/`](02-statistics/) — the most rigorous project here, and a
  real causal-inference design rather than a prediction exercise.
- `05-algorithms-from-scratch/linear-models/01-` and `05-` — clean, working,
  and the source of the tested implementations in
  [`src/ml_from_scratch/`](../src/ml_from_scratch/).

**Also worth reading, for a different reason:**

- `05-algorithms-from-scratch/linear-models/07-mlp-backprop-from-scratch.ipynb`
  — hand-derived backprop, **verified against central differences to 4e-10**.
  A gradient nobody checked is a guess; this one is not.
- `05-algorithms-from-scratch/tree-models/01-decision-tree-information-gain.ipynb`
  — carries a regression test for the `and col` bug, and shows the
  `min_samples_leaf` bias-variance curve.

**Start here if you want the engineering story:** [`BUGFIXES.md`](../BUGFIXES.md).

## Note on provenance

Most of these originated as coursework projects, then were extended and
rewritten. The distinction matters when reading conclusions: the bank-debt and
song-genre projects contain genuine added analysis (regression discontinuity
modelling, class rebalancing), while others are closer to following a guided
template.

Two projects also carry a methodological caveat in their own output, not just in
this file, because the caveat is part of the result:

- **NYSIIS name gender** — the phonetic join is lossy (`beach` and `bitch`
  collapse to the same key) and gender inference from names is unreliable and
  ethically fraught. Kept for the string mechanics.
- **Synthetic stand-ins** — wherever a real dataset is unavailable, the numbers
  describe generated shapes.
