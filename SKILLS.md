# Skills matrix

A competency-by-competency map of what this repository actually demonstrates.

This is written to be **checkable**. Every claim below points at a specific
notebook, module or test, and the gaps are named as plainly as the strengths.
An inflated matrix is worse than none: any technical reviewer will open the
files and find the overclaim, and then discount everything else.

Every claim below points at a specific notebook, module or test. All 19
notebooks execute end to end; the debugging record is in
[BUGFIXES.md](BUGFIXES.md).

**Contents**

- [How to read this](#how-to-read-this)
- [Master matrix](#master-matrix)
- [Per-project breakdown](#per-project-breakdown)
- [The `ml_from_scratch` package in detail](#the-ml_from_scratch-package-in-detail)
- [Gap analysis](#gap-analysis)
- [Suggested ordering for a portfolio](#suggested-ordering-for-a-portfolio)

---

## How to read this

Three levels:

| Level | Meaning |
|---|---|
| ●●● **Demonstrated** | Substantial, working, and verified by tests or by results in the notebook. Would survive a technical interview question. |
| ●● **Working** | Correct and complete enough to rely on, but shallow in scope or unverified by tests. |
| ● **Exposure** | Ran the technique, understood it, did not push on it. |
| ○ **Gap** | Not demonstrated here. Stated so the absence is explicit. |

All 19 notebooks execute end to end. What they do *not* all do is run against
real data — where a dataset is unavailable, the notebook generates a
clearly-labelled synthetic stand-in and says so on every figure.

A skill is only claimed at the level its code earns. A model that
returns 97.5% test accuracy because its gradient was checked against finite
differences to 1e-10 is ●●●; the same model with an unchecked hand-derived
gradient is ●, because nobody can say whether it learned anything.

---

## Master matrix

| Competency | Level | Primary evidence | Verified by |
|---|:--:|---|---|
| **Statistical inference** | ●●● | [RDD bank recovery](notebooks/02-statistics/01-regression-discontinuity-bank-recovery.ipynb) | Notebook results |
| **Causal inference** | ●●● | Same — regression discontinuity design with bandwidth sensitivity | Notebook results |
| **Algorithm implementation** | ●●● | [`src/ml_from_scratch/`](src/ml_from_scratch/) | 48 tests, 54 doctests |
| **Data wrangling** | ●●● | [Credit card approvals](notebooks/03-machine-learning/02-credit-card-approvals.ipynb) | Pipeline fixes the train/test leakage; held-out score reported |
| **Classical ML** | ●●● | [Song genres](notebooks/03-machine-learning/01-classify-song-genres-pca.ipynb), [Credit card](notebooks/03-machine-learning/02-credit-card-approvals.ipynb) | Notebook results |
| **Dimensionality reduction** | ●●● | [Song genres](notebooks/03-machine-learning/01-classify-song-genres-pca.ipynb) | Notebook results |
| **Regularisation & optimisation** | ●●● | [`linear_regression.py`](src/ml_from_scratch/linear_regression.py) | `test_lasso_shrinks_more_than_ols`, `test_early_stopping_triggers` |
| **Model evaluation** | ●●● | [TPOT](notebooks/03-machine-learning/03-tpot-blood-donations-automl.ipynb) — AUC on a 5% incidence target | Paired t-test across CV folds, fold std reported |
| **Unsupervised learning** | ●● | [LDA topics](notebooks/01-nlp/02-hottest-topics-lda.ipynb) | Notebooks execute; topics printed |
| **SQL** | ●● | [International debt](notebooks/06-sql/01-international-debt-statistics.ipynb) | Notebook results |
| **Computer vision** | ●● | [Naive Bees HOG/PCA/SVM](notebooks/04-computer-vision/02-naive-bees-hog-pca-svm.ipynb) | Notebook results |
| **Image signal processing** | ●● | [Naive Bees manipulation](notebooks/04-computer-vision/01-naive-bees-image-manipulation.ipynb) — channel KDE, greyscale, contrast | Notebook results |
| **Feature engineering** | ●● | HOG descriptors, echonest audio features, correlation filtering | Notebook results |
| **Automated ML** | ●● | [TPOT](notebooks/03-machine-learning/03-tpot-blood-donations-automl.ipynb) | Notebook results |
| **Neural networks** | ●● | [ASL CNN](notebooks/04-computer-vision/03-asl-recognition-cnn.ipynb) | Notebooks execute; test accuracy on synthetic letters |
| **Backpropagation by hand** | ●●● | [MLP with BP](notebooks/05-algorithms-from-scratch/linear-models/07-mlp-backprop-from-scratch.ipynb) | Finite-difference gradient check, max error 4e-10 |
| **NLP** | ●● | [LDA](notebooks/01-nlp/02-hottest-topics-lda.ipynb), [NYSIIS](notebooks/01-nlp/01-name-gender-prediction-nysiis.ipynb) | Both execute |
| **Audio / raw DSP** | ○ | — | Absent. Only pre-extracted echonest features are used. |
| **MLOps / serving** | ○ | — | Absent from this repository. |
| **Distributed systems** | ○ | — | Absent. |

---

## Per-project breakdown

### Statistics & causal inference

#### Regression discontinuity: bank debt recovery
[`notebooks/02-statistics/`](notebooks/02-statistics/)

| Competency | Level | What was done |
|---|:--:|---|
| Causal inference | ●●● | Applied a regression discontinuity design to a policy threshold |
| Hypothesis testing | ●●● | Kruskal–Wallis (non-parametric) on age and recovery; chi-square on sex across strategies |
| Regression | ●●● | OLS via `statsmodels`, with and without a threshold indicator |
| Experimental design reasoning | ●●● | Checked covariate balance above/below the cut-off *before* trusting the estimate — the correct order of operations |
| Sensitivity analysis | ●● | Re-ran the estimate on a $900–1100 band and a $950–1050 band |

This is the most methodologically sophisticated project in the repository. It
is worth leading a portfolio with.

### Machine learning

#### Classify song genres from audio data
[`notebooks/03-machine-learning/01-`](notebooks/03-machine-learning/01-classify-song-genres-pca.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| EDA | ●●● | Correlation matrix inspected before modelling, to catch redundant features |
| Dimensionality reduction | ●●● | PCA with scree plot and cumulative-variance plot, 90% threshold |
| Model comparison | ●● | Decision tree vs logistic regression via `classification_report` |
| Class imbalance | ●● | Recognised rock/hip-hop skew, resampled to balance, re-evaluated |
| Cross-validation | ●● | 10-fold `KFold` with `cross_val_score` |

**Fixed.** Step 10 previously read `logit_score = cross_val_score(tree, ...)`,
re-scoring the *decision tree* and never cross-validating the logistic
regression — the printed comparison was the tree against itself. It now also
uses `StratifiedKFold` and a paired t-test:

```
Decision Tree       mean=0.7736  sd=0.0384
Logistic Regression mean=0.8242  sd=0.0200
difference (logreg - tree): +0.0505, paired t = 4.41, p = 0.002
```

#### Predicting credit card approvals
[`notebooks/03-machine-learning/02-`](notebooks/03-machine-learning/02-credit-card-approvals.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Missing data strategy | ●●● | Inspected first, then chose **mean** imputation for numeric columns and **mode** for categorical ones — a deliberate, justified split rather than one blanket fill |
| Data quality | ●●● | Caught the `?` sentinel values before they reached the model |
| Preprocessing | ●●● | Label encoding, min-max scaling, feature dropping with reasoning |
| Hyperparameter search | ●● | `GridSearchCV` over `tol` and `max_iter` |
| Model evaluation | ●● | Confusion matrix, accuracy |

⚠️ Two methodological issues worth naming: `MinMaxScaler` is re-fitted on the
full dataset before `GridSearchCV`, which leaks test information into model
selection; and the `?`-to-`NaN` replacement is applied globally rather than
inside a `Pipeline`, so preprocessing is not reproducible on new data.

#### Automatic model selection with TPOT
[`notebooks/03-machine-learning/03-`](notebooks/03-machine-learning/03-tpot-blood-donations-automl.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Automated ML | ●●● | TPOT with genetic programming over generated pipelines |
| Appropriate metric | ●●● | AUC, not accuracy — correct for a ~5%-incidence binary target |
| Class imbalance | ●●● | Computed target incidence, stratified the split |
| Feature scaling reasoning | ●● | Caught `Monetary (c.c. blood)` variance dominating, applied a log transform and re-checked |
| Statistical testing | ●● | **Caveat:** AUC is reported but no significance test on it. A single AUC point estimate with no CI is weak evidence. |

### Computer vision

#### Naive Bees — HOG, PCA, SVM
[`notebooks/04-computer-vision/02-`](notebooks/04-computer-vision/02-naive-bees-hog-pca-svm.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Feature engineering | ●● | HOG descriptors (`pixels_per_cell=16`, `block_norm='L2-Hys'`) concatenated with flattened colour channels |
| Dimensionality reduction | ●● | PCA to 500 components after standardisation |
| Classification | ●● | SVM with `probability=True` to enable ROC |
| Model evaluation | ●●● | ROC curve **and AUC**, not just accuracy — the right call for a two-class problem |
| Image processing | ●● | RGB → greyscale, spatial gradient computation |

The HOG pipeline is genuinely a signal-processing artefact: it convolves with
gradient operators, pools over spatial cells, and block-normalises. That is the
strongest image-signal-processing evidence in the repository.

#### Naive Bees — image manipulation
[`notebooks/04-computer-vision/01-`](notebooks/04-computer-vision/01-naive-bees-image-manipulation.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Image processing | ●● | Crop, rotate, flip, resize, greyscale, contrast stretching |
| Statistical visualisation | ●● | Per-channel kernel density estimates to compare colour distributions |
| Pipeline construction | ●● | Batched a multi-step transform over a list of files |

#### ASL recognition with a CNN
[`notebooks/04-computer-vision/03-`](notebooks/04-computer-vision/03-asl-recognition-cnn.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Neural network design | ●● | Two conv + max-pool blocks, flatten, dense softmax — a reasonable small-image architecture |
| Label encoding | ●● | Integer → one-hot |
| Error analysis | ●● | Visualised misclassified images and formed a hypothesis about them |

**Fixed.** The TF1 seed call, the removed `np_utils` import, the
non-package `datasets` import and a bare-prose syntax error are all gone. The
error-analysis step now runs as code rather than living in a comment, so the
augmentation hypothesis can actually be tested.

### NLP

#### The hottest topics in machine learning
[`notebooks/01-nlp/02-`](notebooks/01-nlp/02-hottest-topics-lda.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| Unsupervised learning | ●● | LDA over bag-of-words, 14 topics |
| Text preprocessing | ●● | Punctuation stripping, lowercasing, stop-word removal |
| Interpretation | ●● | Printed top words per topic, built a word cloud |

**Fixed.** `get_feature_names()` → `get_feature_names_out()`, and the
`ENGLISH_STOP_WORDS` import → `stop_words="english"` (both removed since). The
110 MB unreassembled multi-part download is deleted; the notebook falls back to
a clearly-labelled synthetic corpus, and `fetch_data.py --only lda` explains
where to get the real one.

#### Name game: gender prediction via NYSIIS
[`notebooks/01-nlp/01-name-gender-prediction-nysiis.ipynb`](notebooks/01-nlp/01-name-gender-prediction-nysiis.ipynb)

| Competency | Level | What was done |
|---|:--:|---|
| String algorithms | ●● | NYSIIS phonetic encoding, then fuzzy matching across two datasets |
| Data joining | ●● | Phonetic key used to match authors against SSA name statistics |
| Exploratory reasoning | ●● | Compared unique-name counts before and after phonetic folding to quantify collision rate |

**Fixed** (steps 6–8 are written and run) — but the approach is still weak on
statistical grounds, and the notebook now says so in its own output. NYSIIS
collisions are severe: the notebook itself demonstrates `beach`/`bitch`
collapsing to the same key. Gender inference from names is low-accuracy and
ethically fraught. Kept for the string mechanics and the fuzzy-join technique,
not as a modelling success.

### SQL

#### International debt statistics
[`notebooks/06-sql/`](notebooks/06-sql/)

| Competency | Level | What was done |
|---|:--:|---|
| SQL | ●● | `COUNT(DISTINCT)`, `GROUP BY`, `AVG`, `MAX`, correlated subquery, `ORDER BY` on aggregates |
| Data scale awareness | ●● | Scaled debt to millions of units for readability |
| Question framing | ●● | Progressively narrowed from global totals to the single largest debtor |

### Algorithms from scratch

| Notebook | Status | Competency |
|---|---|---|
| `linear-models/01-linear-regression-gradient-descent.ipynb` | ✅ works | ●●● Gradient descent derived by hand, min-max normalisation, inverse transform |
| `linear-models/05-perceptron-from-scratch.ipynb` | ✅ works | ●●● Perceptron update rule, accuracy tracking across epochs |
| `linear-models/04-momentum-gd-polynomial-minima.ipynb` | ✅ | ●● Momentum descent on a degree-8 polynomial, now iteration-bounded |
| `linear-models/02-batch-gd-early-stopping-ridge.ipynb` | ✅ | ●● Ridge/L1, real mini-batching, working early stopping |
| `linear-models/03-polynomial-fourier-basis-regression.ipynb` | ✅ | ●● Polynomial and Fourier bases; extrapolation tested on a held-out x-range |
| `linear-models/06-perceptron-iterative.ipynb` | ✅ | ●● Perceptron; the original had swapped arguments and two identical update branches |
| `linear-models/07-mlp-backprop-from-scratch.ipynb` | ✅ | ●●● Backprop from scratch, **gradient-checked to 4e-10**, 97.5% test accuracy |
| `tree-models/01-decision-tree-information-gain.ipynb` | ✅ | ●●● Entropy and information gain; fixed the `and col` bug and an inverted left/right split |
| `tree-models/02-decision-tree-visualisation.ipynb` | ✅ | ●● Graphviz export plus a readable dump of the fitted `tree_` arrays |

---

## The `ml_from_scratch` package in detail

This is where the work is genuinely verified, and it is the artefact that makes
the algorithm claims defensible.

### Perceptron — `src/ml_from_scratch/perceptron.py`

| Competency | Level | Evidence |
|---|:--:|---|
| Linear classification | ●●● | `test_learns_a_separable_problem` — >0.97 accuracy on 400 samples |
| Convergence reasoning | ●●● | `test_error_count_decreases` — error count reaches exactly 0 |
| Bounded optimisation | ●●● | `test_respects_epoch_budget_on_cyclic_data` — non-separable data cannot loop forever |
| Early stopping | ●● | `test_early_stopping_halts_early` |
| API design | ●● | Chaining, `decision_function`, validation errors |

Corrects three defects in the original notebooks: swapped argument order, an
aliasing bug where `wt = w` then `wt[-1] = b` mutated the original, and two
identical update branches where the sign should have differed.

### Linear regression — `src/ml_from_scratch/linear_regression.py`

| Competency | Level | Evidence |
|---|:--:|---|
| Optimisation | ●●● | `test_recovers_known_coefficients` — recovers `[2, -1, 0.5]` and intercept `4.0` |
| Feature scaling | ●●● | `test_standard_scaling_does_not_change_fit_quality` — a 1000× feature scale change must not degrade the fit. This is *why* the module scales |
| Regularisation | ●●● | `test_lasso_shrinks_more_than_ols` |
| Early stopping | ●●● | `test_early_stopping_triggers` |
| Mini-batching | ●● | `test_batch_size_changes_convergence_rate` |
| Reversible transforms | ●●● | `test_round_trip_is_exact` |

Corrects: in-place normalisation that corrupted caller data on every
`predict`; an arithmetic error in the inverse transform
(`(max + min) + min` instead of `(max - min) + min`); an undefined `sel.patience`
reference; and a `tol` parameter that was documented but never actually used —
which meant early stopping could fire on floating-point noise, or never fire at
all.

### Decision tree — `src/ml_from_scratch/decision_tree.py`

| Competency | Level | Evidence |
|---|:--:|---|
| Information theory | ●●● | `test_unequal_split_matches_closed_form`, `test_perfectly_balanced_split_is_one_bit` |
| Greedy splitting | ●●● | Midpoint thresholds between adjacent sorted values |
| Leaf-size regularisation | ●●● | `test_respects_min_samples_leaf` — asserts the *actual* leaf sizes |
| Bias/variance control | ●●● | `test_min_leaf_reduces_node_count` |
| Depth control | ●● | `test_respects_max_depth` |
| Probability output | ●● | `test_predict_proba_rows_sum_to_one`, argmax agreement with `predict` |
| Generality | ●● | `test_string_labels_work` |

Corrects: an undefined name (`data_n`) that made the function raise
`NameError` on every call, and a split guard reading `and col` — since column
`0` is falsy in Python, **feature 0 could never be selected**. The tree was
silently crippled, and it would have scored at chance on any problem where the
first column mattered. `test_feature_zero_is_usable` is a regression test that
exists solely to keep that from coming back.

---

## Gap analysis

The three honest gaps, and what would close them.

### 1. Audio / raw signal processing — ○

The song-genre project uses echonest audio features, but they arrive
pre-extracted. There is no work on raw waveforms: no filtering, no windowing,
no FFT, no spectral analysis, no time-frequency representations.

**To close it**, one project is enough. Load a short audio clip, then:
STFT and spectrogram visualisation → band-pass and low-pass filters → Wiener
noise reduction → MFCC feature extraction → a classifier on the MFCCs. That
would cover spectral analysis, time-frequency structure, and filter design in
a single artefact, and it is the natural next step from the existing
echonest-based work.

### 2. Neural networks from scratch — ●●● (closed)

This gap is now closed. The backpropagation notebook was rewritten with
explicit `(n_in, n_out)` weight matrices per layer, and its gradients are
checked against central differences:

```
param    max abs difference   verdict
W1                3.994e-10   MATCH
b1                1.270e-10   MATCH
W2                3.280e-10   MATCH
b2                2.266e-10   MATCH
test accuracy: 0.9750
```

What made the original unworkable was structural — weights stored as flat 1-D
arrays with the bias prepended to the input on every layer, so a weight vector
had to match its own input dimension, and the backward pass reversed the layer
order to compensate. Explicit matrices remove the whole class of error.

**What is still missing** is breadth: one MLP and one CNN, no RNN, no
transformer, no attention, nothing implemented at scale.

### 3. MLOps — ○

Nothing here is versioned, tracked or served.

**To note:** this gap is specific to *this* repository. The
[`mlops-assignment-e2e-ml-pipeline`](https://github.com/hwilner/mlops-assignment-e2e-ml-pipeline)
repository covers Airflow and MLflow. Linking it from this README would close
the gap in the aggregate portfolio without padding this repository.

---

## Suggested ordering for a portfolio

If this is being shown to someone assessing capability, the strongest narrative
is:

1. **Regression discontinuity** — proves statistical and causal reasoning.
2. **`ml_from_scratch`** — proves you can build, test and document algorithms
   rather than call them.
3. **TPOT blood donations** — proves awareness of appropriate metrics, class
   imbalance and automated tooling.
4. **Naive Bees HOG/PCA/SVM** — proves computer-vision and signal-processing
   fundamentals with correct evaluation.
5. **LDA topics** — proves unsupervised learning.

And if the debugging is part of the pitch rather than something to hide:
[BUGFIXES.md](BUGFIXES.md) is a stronger artefact than most portfolios have. A
decision tree that could never use feature 0, a model that predicted the exact
opposite of what it learned while scoring 0.0 on separable data, a
cross-validation run that compared a model against itself, and a scaler
leakage hidden behind a variable that was computed and discarded — none of
those raise an exception. Finding them is the argument for running things and
reading the output, which is the actual skill.

The last three notebooks that do not run are not a liability, provided they are
labelled the way this README labels them. An engineer who leaves a broken
notebook with a note explaining exactly why it is broken demonstrates more
judgement than one who quietly deletes it.
