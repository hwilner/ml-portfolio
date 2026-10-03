# Retrospective: `small--projects`, 2019 → 2026

**What this is.** A three-column review of every project in this repository.
The columns are deliberately distinct, because the distinction between *"I
didn't know better"* and *"the tool didn't exist yet"* is the whole point:

| Column | Tools available | Knowledge applied |
|---|---|---|
| **1. 2019, as written** | 2019 | 2019 |
| **2. 2019, redone** | **2019** | 2026 |
| **3. 2026** | **2026** | 2026 |

Column 2 is the one that actually teaches. A project can fail for reasons the
field had already solved (column 2), or because the material needed did not yet
exist (column 3). Most of the defects in this repository are the first kind —
which is worth saying plainly, because it is the more fixable of the two.

A prior external review of this repository fed into this document. Where it
was right, it was usually right about things already fixed during the debugging
pass; where it was wrong, it was wrong because it read the repository before
that pass and could not execute the notebooks. Both are recorded.

---

## Part 0 — The premise changed while the review was being written

The external review describes a repository where six notebooks do not run, one
is unfinished, and the decision tree and cross-validation contain live defects.
That was accurate when written. It is no longer.

```
19 notebooks    19 executing    0 failing
48 package tests + 54 doctests + 4 module doctests    0 failing
```

This matters for the retrospective, because a large share of the review's
"what could have been better even then" list was in fact **already fixed**. The
honest accounting:

| Review said | Reality now |
|---|---|
| MLP backprop "genuinely unfinished", "the biggest missed opportunity" | Rewritten with explicit weight matrices; **gradient-checked to 4e-10**; 97.5% test accuracy |
| ASL notebook "does not run" | Runs; TF2 APIs, loader and syntax error all fixed |
| NYSIIS steps 6–8 "still unfilled placeholders" | Written and running |
| Decision tree `and col` disables feature 0 | Fixed, with a regression test named for the bug |
| Song genres scores the tree twice | Fixed; real result 0.824 vs 0.774, p = 0.002 |
| Credit card leaks the scaler into model selection | Fixed with a `Pipeline`; held-out score now reported |
| Ridge notebook raises `NameError` | Rewritten; real mini-batching and working early stopping |
| LDA uses a removed scikit-learn API | Fixed (`get_feature_names_out`, `stop_words="english"`) |

So the remaining gaps in column 2 are much smaller than the review implies.
What is genuinely still undone is listed in **Part 4** — and it is almost
entirely column 3 material, plus engineering infrastructure.

---

# Part 1 — Statistics & causal inference

## Regression discontinuity: bank debt recovery
`notebooks/02-statistics/01-regression-discontinuity-bank-recovery.ipynb`

**The strongest project in the repository**, before and after. It asks a causal
question rather than a predictive one, and it tests the design's assumptions
before trusting the estimate.

### 1. 2019, as written

Load the data, scatter-plot around the $1,000 threshold, run a Kruskal–Wallis
test on age across recovery strategies and a chi-square on sex, then fit OLS
with a threshold indicator and repeat on a narrower band. The conclusion is
that crossing the threshold changes actual recovery.

The reasoning order is right, and that is rare: **balance checks first, effect
estimate second**. Many working portfolios do it the other way round.

### 2. 2019, redone

Everything below was possible in 2019 with `statsmodels`, `scipy` and
matplotlib. None of it is a modern-tooling suggestion.

- **State the estimand before estimating it.** "The average effect on recovered
  debt of crossing the $1,000 threshold" is a sentence. Without it, the
  regression coefficient is a number whose meaning is inferred after the fact.
- **Fit separate slopes either side of the cut-off**, rather than one line plus
  an indicator. A single slope assumes the relationship has the same gradient on
  both sides, which is an assumption about the *functional form* and is
  separate from the continuity assumption the design actually needs.
- **Use heteroskedasticity-robust standard errors** (`cov_type="HC3"`). Bank
  debt is right-skewed by a long way; homoskedastic OLS standard errors are
  wrong here, and the fix is one keyword in 2019.
- **Run a manipulation test.** This is the question the whole design turns on:
  could an account owner arrange to land just below $1,000? The McCrary
  density test at the cut-off answers it, and it is a density plot — no exotic
  tooling. The reason the bank collects less after the threshold is precisely
  the reason a debtor might try to stay under it.
- **Placebo cut-offs.** Re-run at $1,500 and $2,000. If a "significant" effect
  appears at thresholds where no policy changes, the estimator is finding
  something other than the policy. This is the cheapest available refutation
  and it was available in 2019.
- **Choose the bandwidth by a rule, not by hand.** Two windows is a
  sensitivity check; a stated selection criterion is an estimate. Even in
  2019, leave-one-out cross-validation over candidate bandwidths was
  implementable in a few lines.
- **Plot the fitted functions with confidence bands around the cut-off.** A
  reader should be able to see the discontinuity, not infer it from a
  coefficient.

### 3. 2026

The conceptual work in column 2 is unchanged — it was always the hard part. What
2026 adds is tooling that makes the *refutation* systematic rather than
remembered:

- **`dowhy`** formalises model → identify → estimate → refute, and ships RD
  estimators. Its value is not the estimate; it is that the identification
  assumptions get written down where they can be argued with.
- **Bias-corrected inference and automated bandwidth selection** (`rdrobust`,
  `rddensity`) replace hand-picked windows, and give robust confidence
  intervals rather than OLS asymptotic ones.
- **`rdensity`** makes the manipulation test a one-liner.
- A **local-linear-quadratic** fit with a data-driven bandwidth, reported
  alongside the simple OLS result so the transparency of the original is not
  lost.

**Recommendation:** this is the project to modernise first. It is the one a
statistically literate reviewer will open first, and it is currently the
furthest from what a 2026 statistical audience expects.

---

# Part 2 — Classical machine learning

## Classifying song genres from audio features
`notebooks/03-machine-learning/01-classify-song-genres-pca.ipynb`

### 1. 2019, as written

Inspect the correlation matrix, standard-scale, PCA to 90% cumulative variance,
train a decision tree and a logistic regression, notice the rock/hip-hop
imbalance, resample to balance, re-evaluate, then 10-fold cross-validation.

The conceptual sequence is good — EDA before modelling, dimensionality
reduction justified by a plot rather than a rule of thumb, imbalance noticed.

Two defects, now fixed: the cross-validation compared the decision tree against
*itself*, and the final conclusion rested on that comparison.

### 2. 2019, redone

- **Put everything in one `Pipeline`**: `StandardScaler → PCA → estimator`. The
  scaler and PCA were fitted once on all the data and then treated as fixed
  during cross-validation, which leaks. `sklearn.pipeline` has existed since
  0.14; this was a 2019 fix.
- **Move PCA's component count inside model selection.** Choosing `n_components`
  by looking at the whole dataset and then cross-validating only the classifier
  is a second, subtler leak.
- **Do not rebalance before splitting.** Downsampling to the minority class
  throws away ~3,000 majority rows *and* was done before the train/test split,
  so the same track can appear on both sides. Resample inside the fold, or
  better, use `class_weight="balanced"`, which keeps all the data.
- **Add a non-PCA baseline.** If PCA did not help, the honest result is that it
  did not help, and that is worth knowing.
- **Report PR-AUC alongside ROC-AUC**, and balanced accuracy — with a 4:1
  imbalance, accuracy is the one number guaranteed to flatter you.
- **Keep a truly untouched final test set.** Hyperparameters were being read off
  cross-validation results in the same notebook that reported them.

### 3. 2026

- **The real gap is audio, not classification.** The echonest features arrive
  pre-extracted, so this is "model some audio-derived numbers". The modern
  version computes its own representation: STFT → spectrogram → MFCCs →
  classifier. That single change moves the project from applied statistics into
  signal processing, which is the competency the repository most needs.
- **Compare handcrafted against learned features explicitly.** Echonest-style
  features vs a pretrained audio encoder (CLAP audio embeddings, or a small
  audio transformer), same classifier, same protocol. The comparison is the
  artefact.
- **Modern tabular baselines** — `HistGradientBoosting`, then LightGBM/CatBoost.
  Logistic regression is a baseline, not a competitor, on tabular data in 2026.
- **BPE / TPE search** instead of a hand-built grid, with the budget stated.
- **Calibration curves**, because a genre classifier gets used with thresholds.

## Predicting credit card approvals
`notebooks/03-machine-learning/02-credit-card-approvals.ipynb`

### 1. 2019, as written

Inspect, catch the `?` sentinel, impute — mean for numeric, mode for
categorical — label-encode, drop two low-value features, scale, fit logistic
regression, grid-search `tol` and `max_iter`, report accuracy and a confusion
matrix.

The imputation split by column type is the best piece of data thinking here,
and it was the right call on its own merits.

### 2. 2019, redone

- **The scaler leak.** Scaling was refitted on the full dataset, and the scaled
  array was then never used — the search ran on raw data. A `Pipeline` fixes
  both. This is a 2019 fix.
- **`OneHotEncoder` instead of `LabelEncoder`** for the six nominal variables.
  Arbitrary integer codes impose a false ordering on `Married` and `Education`
  that the model will happily exploit.
- **Grid-search `C`**, not `max_iter` and `tol`. The default `C=1.0` was never
  examined, and regularisation strength is the hyperparameter that actually
  matters for logistic regression. Tuning convergence tolerances is tuning
  whether the solver finished, not how good the model is.
- **Stratified split.** The classes are imbalanced; a plain split makes the
  test-set class ratio a coin flip.
- **Report PR-AUC and calibration**, and pick a decision threshold from a
  stated false-positive/false-negative cost rather than at 0.5. In credit
  underwriting the costs are wildly asymmetric.
- **Hold out a final test set** distinct from the CV folds.

The honest reporting that came out of fixing the leak is the interesting part:

```
Best: 0.852174 using {'logreg__max_iter': 100, 'logreg__tol': 0.01}
fold-to-fold std at the best config: 0.1372
tuned pipeline on the held-out test split: 0.8296 (+/- 0.0613)
```

CV claimed 0.852; the model scores **0.830** on held-out data, with a
fold-to-fold standard deviation of **0.137**. Most of the tuning gain was
noise, and the original bug concealed it.

### 3. 2026

- **`ColumnTransformer`** with parallel numeric and categorical branches:
  `SimpleImputer` → `StandardScaler`, and `SimpleImputer` → `OneHotEncoder`.
  This is the standard answer and it was available in 2019.
- **A real baseline ladder** — logistic, then gradient boosting (CatBoost
  handles the categorical columns natively and needs no encoding).
- **Optuna** with a stated time budget, rather than a 3×3 grid over solver
  settings.
- **Repeated stratified CV**, and **nested CV** if the model selection itself
  needs an unbiased estimate.
- **Calibration**, and a threshold-cost analysis.
- **A fairness section, and it should not be optional.** This is a lending
  decision. Protected attributes are in the feature set (`Sex`, `Married`,
  `Ethnicity`, `Age`, `ZipCode`). A 2026 version that does not examine disparate
  impact and per-group calibration is incomplete in a way that a 2019 version
  was not — expectations moved, and the stakes are real.
- **SHAP values** for per-prediction explanation.

**Recommendation:** the best candidate for a full modern rewrite. It is
tabular, the machinery is well established, and the fairness dimension is
genuinely worth demonstrating.

## Automatic model selection with TPOT
`notebooks/03-machine-learning/03-tpot-blood-donations-automl.ipynb`

### 1. 2019, as written

Check target incidence, stratify the split, hand the pipeline search to TPOT,
score with AUC, notice `Monetary (c.c. blood)` dominates the variance, log
transform it, and compare against a hand-built logistic regression.

Choosing AUC over accuracy on a 5%-incidence target is the decision that
matters most in this notebook, and it was made correctly.

### 2. 2019, redone

- **A point estimate of AUC is not evidence.** Five folds have a standard
  deviation; report it. When two models differ by 0.01, the fold spread is
  usually larger than the difference.
- **Keep the log transform inside the pipeline**, so it is refitted per fold
  and survives to inference on new data. It was applied as a dataframe
  operation before the split.
- **A time budget, stated.** Genetic programming without a wall-clock limit can
  run for an hour and return a pipeline fitted to noise.
- **Compare AutoML against a serious baseline**, not just logistic regression.
  If AutoML cannot beat a well-tuned gradient booster, that is the finding, and
  it is a much more interesting result than the pipeline it returned.
- **PR-AUC alongside ROC-AUC.** At 5% prevalence, precision is the number that
  moves.

### 3. 2026

- **The framing changes.** In 2019, AutoML was the novelty. In 2026 the
  interesting question is: *can automated search beat a carefully built
  baseline under an identical time and compute budget?* That is an experiment.
  A three-rung ladder — logistic, hand-built gradient boosting, AutoML — with
  time, memory and final score reported for each makes TPOT an instrument
  rather than a demonstration.
- **Optuna (TPE)** for the search, with pruning of clearly-losing trials.
  Genetic programming has largely been superseded for this.
- **AutoGluon** as a strong reference point, since it produces stacked
  ensembles that are hard to beat.
- **Track ROC-AUC, PR-AUC, Brier score, wall-clock and model size** — the last
  two because an AUC that costs 40× the training time and gains 0.005 is not a
  result worth deploying.

---

# Part 3 — Computer vision

## Naive Bees: image manipulation
`notebooks/04-computer-vision/01-naive-bees-image-manipulation.ipynb`

### 1. 2019, as written

Crop, rotate, flip, resize, convert to greyscale, stretch contrast, plot
per-channel kernel density estimates for two images, then batch the
transformations into a loop.

The per-channel KDE comparison is the valuable part: it turns "these look
different" into a measurement, and it is what motivates dropping colour in the
next notebook.

### 2. 2019, redone

- **Pure functions, one image each, plus one loop.** The transforms were inline
  in display cells, so nothing was reusable or testable.
- **Assert shape and dtype invariants** on every output. A transform that
  silently returns the wrong shape is a bug that surfaces three cells later.
- **Move the imports and helper functions above first use.** The original
  notebook had cells referencing `plt` before importing it.
- **Compare contrast stretching against histogram equalisation** — they are
  different operations with different use cases, and the notebook only does
  the first without saying why.
- **Document the interpolation method** for resizing, and state whether
  aspect ratio is preserved. Nearest-neighbour and bilinear give visibly
  different results and neither is "the" default.

### 3. 2026

Keep it short and make it a proper **preprocessing and augmentation module**:
`torchvision.transforms` or Albumentations, deterministic seeds for
validation transforms, augmentation applied to training data only, EXIF
orientation handling made explicit, and unit tests asserting output size and
range. The pedagogical content is unchanged; only the packaging changes.

## Naive Bees: HOG + PCA + SVM
`notebooks/04-computer-vision/02-naive-bees-hog-pca-svm.ipynb`

### 1. 2019, as written

Greyscale → HOG with `pixels_per_cell=(16,16)` and L2-Hys block normalisation →
concatenate with flattened colour → standardise → PCA to 500 components →
`SVC(probability=True)` → ROC and AUC.

This is a legitimate classical pipeline, and the decision to report ROC/AUC
rather than accuracy for a two-class problem is correct and was unusual in
2019.

### 2. 2019, redone

- **Scaling and PCA inside the CV pipeline.** Same leak as everywhere else.
- **Tune HOG and SVM jointly** — cell size, orientations, `C`, `gamma` interact.
  Fixing HOG by eye and then tuning the SVM separately cannot find the joint
  optimum.
- **Ablate the feature sets**: HOG only, colour only, and combined. The
  notebook concatenates them without ever finding out which one is carrying the
  signal. That is the most informative single experiment available here.
- **`probability=True` has a cost** — it runs an internal 5-fold
  cross-validation during fitting to calibrate. Worth stating when the model is
  used at scale.
- **Stratified CV and an untouched test set.**
- **Inspect confident mistakes** specifically, not aggregate accuracy.

### 3. 2026

- **This notebook is now a baseline, and that is its value.** The comparison
  that teaches is three representations on identical splits:
  1. HOG + linear/SVM — handcrafted,
  2. frozen pretrained CNN or ViT embeddings + linear classifier,
  3. fine-tuned pretrained model.

  One notebook, and the seven-year shift in computer vision becomes visible as
  a table rather than a claim.
- **Never delete the HOG version.** It is what makes the improvement legible.
- Add Grad-CAM on the fine-tuned model to show *where* it looks, against the
  HOG visualisation's gradient map. The two images are directly comparable and
  it is a genuinely nice piece of work.
- Robustness checks for background and lighting variation — the failure mode
  that matters for this specific dataset.

## ASL recognition with a CNN
`notebooks/04-computer-vision/03-asl-recognition-cnn.ipynb`

### 1. 2019, as written

Two conv/max-pool blocks, flatten, dense softmax over A/B/C. One-hot labels,
RMSProp, categorical cross-entropy, a validation split, and then a closing
observation that the misclassified images look obviously different to a human —
so the model is probably too simple and needs augmentation.

That closing observation is the best piece of reasoning in the notebook. It
looks at failures, forms a hypothesis, and proposes an intervention. It was,
however, a *comment* rather than a cell, so it could never be checked. It is
now executable code.

### 2. 2019, redone

- **A signer-level split, and this is the important one.** Random image-level
  splitting lets the model learn characteristics of the *photographer* and the
  *background* rather than the letter. If every photo of one signer's "A" lands
  in training and a different signer's "A" lands in test, the reported accuracy
  measures memorisation of a person. The dataset almost certainly supports
  grouping by signer; if it does not, that is a stated limitation.
- **Separate train/validation/test explicitly**, with the test set touched once.
- **Augmentation** — the hypothesis the notebook itself reached. Rotation,
  translation, brightness and scale. The handwriting varies enormously and
  invariance to those transforms is exactly what the model lacks.
- **Early stopping and checkpointing**; report the best epoch, not the last.
- **Per-class recall and a confusion matrix**, not aggregate accuracy. A/B/C
  are visually distinct, so aggregate accuracy hides which confusion dominates.
- **Inspect the specific confusions** — which letters does the model confuse,
  and does that match human confusability?

### 3. 2026

- **Transfer learning, not random initialisation.** A pretrained
  MobileNet/ResNet/EfficientNet/ViT backbone, frozen first, then fine-tuned.
  Training a small CNN from scratch on a few hundred images is no longer the
  natural first move.
- **The historical CNN stays** as the educational baseline — the accuracy gap
  is the point.
- Mixed precision, and **ONNX/TFLite export** if the project is meant to
  demonstrate deployment. Sign-language recognition is a case where on-device
  inference is a real requirement, so this is a substantive addition rather than
  a box-tick.
- Signer-held-out evaluation, which remains the honest metric regardless of
  architecture.

---

# Part 4 — NLP

## "Hottest topics in machine learning" with LDA
`notebooks/01-nlp/02-hottest-topics-lda.ipynb`

### 1. 2019, as written

Count submissions per year, strip punctuation, lowercase, build a word cloud as
a sanity check, `CountVectorizer` with English stop words, LDA with 14 topics,
print the top words per topic.

A complete and reasonable classical topic-modelling exercise for 2019.

### 2. 2019, redone

- **Justify the topic count.** 14 was chosen and not defended. Compare counts,
  and check **stability across random seeds** — a topic model whose topics
  change when you change `random_state` has not found structure. This is
  available in 2019 and is the single most important check on any topic model.
- **Score topic quality.** Coherence or NPMI. "These look like topics" is a
  subjective read; coherence is a measurement.
- **Look at documents, not just words.** Reading the top 5 abstracts per topic
  is what reveals a topic that is a grammatical artefact.
- **Use bigrams**, and lemmatise rather than only stripping punctuation.
- **Separate fitting from interpretation.** Choosing the topic count and then
  reading the topics on the same data is circular.

### 3. 2026

- **Keep LDA as the interpretable baseline, then add a semantic pipeline**:
  sentence-transformer embeddings → UMAP → HDBSCAN or k-means → c-TF-IDF
  descriptions → stability check → evolution over publication year.
- **The payoff is real**: bag-of-words groups documents by shared vocabulary,
  so "deep neural network" and "convolutional network" land in different
  clusters. Embeddings put them together. That is a concrete, demonstrable
  difference, and it makes the NLP shift tangible rather than a slogan.
- **BERTopic** as a packaged version of the same idea, with the caveat that its
  components should be explained rather than treated as magic.
- **Title the honest result.** Even LDA on real NIPS data finds broad,
  slowly-moving themes — depth, sparsity, theory. Presenting that as a finding,
  rather than dressing it up, is worth more.

## Name game: gender prediction via NYSIIS
`notebooks/01-nlp/01-name-gender-prediction-nysiis.ipynb`

### 1. 2019, as written

Encode first names with the New York State Intelligence System algorithm, join
against SSA baby-name statistics on the phonetic key, infer gender, tally by
year, plot.

Steps 6–8 were never written. They are now.

### 2. 2019, redone

- **A phonetic collision is not evidence of identity.** The notebook itself
  shows `beach` and `bitch` collapsing to the same key, and then proceeds as if
  the join were sound. That is the central error, and it was visible at the
  time.
- **Measure the collision rate and the unmatched rate explicitly**, and report
  them before any conclusion.
- **Hand-check a sample by hand.** Twenty names, checked against reality, is
  an afternoon's work and would have falsified the join.
- **Allow ambiguity to stay ambiguous.** Names whose key matches both a
  strongly-female and a strongly-male entry should be `Unknown`, not a coin
  flip.
- **Separate two questions the notebook conflates**: how good is the phonetic
  join, and how good is gender inference. They have different failure modes and
  need separate answers.

### 3. 2026

**I would not modernise this one as a gender-prediction project.** The method
is weak in ways that tools do not fix, and the target attribute is sensitive.
Gender is not recoverable from a name; the notebook's own data source (SSA
applications, skewing toward people who applied) makes that worse, and a
deterministic phonetic join assigns wrong labels with no mechanism to catch it.

Two honest options:

**(a) Reframe as entity resolution — recommended.** This is the same technical
work, applied to a task where it is legitimate and demonstrable:
Unicode normalisation → transliteration → Jaro-Winkler and RapidFuzz baselines →
phonetic algorithms as *one feature among several*, not the decision rule →
character n-gram similarity → a learned matcher → precision/recall on a
hand-labelled pair set. That is a real, modern, useful project, and the
original intuition — that exact string matching is insufficient — was right.

**(b) Delete the gender framing.** If gender information is genuinely needed,
the correct 2026 source is self-reported data with explicit unknown handling.
Inferring it from names is not a modelling gap to be closed; it is a proxy that
should not be used.

**The ethical note is not optional.** A portfolio that demonstrates
gender inference from names without flagging the problem is making a statement,
whether intended or not.

---

# Part 5 — Algorithms from scratch

These notebooks earn their place for one reason: they show whether the mechanics
were understood, not just whether an API was recalled. The debugging pass
confirmed the value of that intent and also demonstrated why from-scratch work
needs *more* rigour than notebook work, not less.

**The finding worth stating:** three of the four worst defects in the entire
repository were in this section, and none of them raised an exception. The
decision tree could not use its first feature; the decision tree predicted the
opposite of what it learned; the perceptron had two identical update branches
and so could not learn at all. All three looked like plausible code. All three
are now regression tests.

### 5.1 Linear regression by gradient descent
`01-linear-regression-gradient-descent.ipynb` — worked once two syntax issues
were fixed. 2019 redo: compare against the closed-form OLS solution, gradient-check
the derivative, never mutate caller arrays, plot loss against iteration, test on
data with known coefficients. 2026: verify against scikit-learn *and* JAX
autograd, add property-based tests, benchmark.

### 5.2 Batch GD with early stopping and ridge
`02-...` — did not run (`sel.patience`); now rewritten with real mini-batching.
2019 redo: test that regularisation actually shrinks coefficients, test that
early stopping actually stops, distinguish batch/mini-batch/stochastic
explicitly. 2026: a clean optimiser loop with reproducible shuffling and full
loss history, then SGD vs momentum vs Adam as a teaching extension.

### 5.3 Polynomial and Fourier basis regression
`03-...` — did not run. Beyond the counter and inverse-normalisation bugs, the
design matrix was rebuilt from the *data* inside `predict`, so the model could
not generalise at all. 2019 redo: one consistent design-matrix function, inspect
the condition number, select degree on validation, compare raw against
standardised bases. 2026: `PolynomialFeatures` + Ridge, splines, a GAM,
gradient boosting — with the educational message that **basis choice encodes
assumptions about the function class**.

### 5.4 Momentum gradient descent
`04-...` — worked but the `while` loop had no iteration cap. 2019 redo: every
iterative routine needs a max-iteration bound, a tolerance, NaN/overflow
handling, a history, and multiple initialisations. 2026: a five-way comparison
— GD, momentum, Nesterov, Adam, L-BFGS — on the same surface, which turns a
single notebook into a compact optimisation lesson.

### 5.5 Perceptron (working)
`05-...` — the one that was already sound. 2019 redo: test both separable and
non-separable cases, visualise the decision boundary, compare with
`sklearn.linear_model.Perceptron`, state the convergence assumptions explicitly.
2026: `partial_fit`, scikit-learn estimator compatibility, benchmark against
`SGDClassifier(loss="perceptron")`.

### 5.6 Iterative perceptron (broken)
`06-...` — did not run. Swapped argument order, a `derr` arity mismatch, aliasing
between `wt` and `w`, and two identical update branches so the rule could not
learn. The external review is right that this is not a portfolio project in its
own right. It is, however, now a good **testing lesson**, and the tests that
caught it are the point. Keep it labelled as such rather than deleting it.

### 5.7 MLP with backpropagation
`07-...` — **the biggest single improvement in the repository.**

The original stored each layer's weights as a flat 1-D array and prepended a
bias column to the input on every layer, so a weight vector had to match its own
*input* dimension. That is why the backward pass reversed the layer order, and
why the hidden-layer update had an impossible shape. No patch fixes a
structural problem like that.

Rewritten with explicit `(n_in, n_out)` matrices per layer and separate biases.
The forward pass is `x @ W + b`; the gradient is `delta @ x.T`. Neither the
ordering bug nor the shape bug can occur.

The part that makes this an argument rather than an assertion is the
**finite-difference gradient check**:

```
param    max abs difference   verdict
W1                3.994e-10   MATCH
b1                1.270e-10   MATCH
W2                3.280e-10   MATCH
b2                2.266e-10   MATCH

training accuracy: 0.9979      test accuracy: 0.9750
```

A hand-derived gradient that has not been checked against numerical
differencing is not a result. It is a guess. This is the single most valuable
artefact in the repository for demonstrating that the mechanics are understood
rather than recalled.

**2026 additions:** check gradients against PyTorch autograd as well, add a
XOR test, assert gradient shapes match parameter shapes, assert probabilities
sum to one.

### 5.8 Decision tree from information gain
`tree-models/01-...` — did not run. 2019 redo, which the review gets exactly
right: test entropy of known distributions, information gain on a
hand-calculated split, **feature 0 as the only informative feature**, minimum
leaf size, maximum depth, string labels, pure-node stopping, constant features.

That third test exists now, named `test_feature_zero_is_usable`, because the
original guard `if ent < min_avg_ent and col:` made feature 0 permanently
ineligible for any dataset. 2026: feature importance, cost-complexity pruning,
categorical splits, scikit-learn API compatibility.

### 5.9 Decision-tree visualisation
`tree-models/02-...` — used `sklearn.externals.six`, removed in scikit-learn
0.23. Now `io.StringIO`, plus a readable dump of `model.tree_.children_left` /
`.feature` / `.threshold`, which is a better teaching artefact than a PNG.
2026: `plot_tree` for a quick look, `export_graphviz` for the full structure,
and permutation importance or SHAP to go beyond structure to behaviour.

---

# Part 6 — SQL

## International debt statistics
`notebooks/06-sql/01-international-debt-statistics.ipynb`

### 1. 2019, as written

`COUNT(DISTINCT)`, `SELECT DISTINCT`, `SUM`/`ROUND`, `GROUP BY` + `ORDER BY` +
`LIMIT 1`, `AVG` grouped by indicator, a correlated subquery against `MAX`, and
a two-column `GROUP BY`. The progression — scope, then totals, then drill into
the dominant category, then the single largest record — is a reasonable shape
for exploratory analysis, and dividing into millions before display is a small
touch that matters.

### 2. 2019, redone

- **A `CREATE TABLE` and schema documentation**, so the notebook is
  self-contained rather than assuming a database someone else built.
- **Window functions** — `SUM(...) OVER (PARTITION BY country)`, `RANK() OVER
  (ORDER BY ...)` — which is the actual 2019 answer to "total per country" and
  "rank the countries", and avoids a self-join.
- **CTEs** to name intermediate steps, so a nine-line query with nested
  subqueries becomes readable.
- **A join across at least two tables.** Everything here is one flat table, so
  the join skill is not demonstrated at all.
- **Null handling and unit assumptions stated explicitly.** Debt is recorded in
  mixed units; `SUM(debt)` without a unit is a claim, not a number.
- **A reproducible database initialisation script**, and **tests asserting the
  known answer** to at least one query — 124 countries is a checkable fact.

### 3. 2026

- **DuckDB queries the CSV directly.** No server, no install, no
  `international_debt.zip`, and it runs in the notebook. For an analytics
  portfolio this is a large usability win and it is the single biggest change
  to this project.
- **Pull the World Bank data programmatically** rather than shipping a zip, so
  the pipeline is reproducible end to end.
- **Store to Parquet**, then query — which introduces partitioning and columnar
  storage without needing a warehouse.
- **dbt-style data tests** (uniqueness, not-null, referential integrity) if the
  project should read as analytics engineering rather than SQL practice.

---

# Part 7 — Cross-cutting themes

## 7.1 What was already solved by 2019 and I did not use

This is the honest core of the retrospective, and it is uncomfortable.

| Technique | Available since | What I did instead |
|---|---|---|
| `Pipeline` | sklearn 0.14 (**Aug 2013**) | Manual, order-dependent preprocessing |
| `ColumnTransformer` | sklearn 0.20 (**Nov 2018**) | Per-column `for` loops |
| `OneHotEncoder` | always | `LabelEncoder` on nominal columns |
| `StratifiedKFold` | sklearn 0.10 (2012) | `KFold` on imbalanced data |
| `StratifiedShuffleSplit`, `RepeatedStratifiedKFold` | sklearn 0.14 / 0.16 | one plain split |
| Group-aware splitting (`GroupKFold`, `StratifiedGroupKFold`) | sklearn 0.17 (2017) | random image-level split on ASL |
| ROC-AUC / PR-AUC | sklearn 0.9 (2014) | used in some projects, accuracy in others |
| `cross_val_score` with a `Pipeline` | sklearn 0.14 (2013) | CV on a pre-transformed array |
| `np.random.default_rng` | NumPy 1.17 (**Jul 2019**) | legacy `np.random.seed` — a boundary case, it shipped that same year |
| Matplotlib inline | matplotlib 1.4 (2015) | `%matplotlib inline` scattered mid-notebook |
| `os.path.join` | always | hard-coded `"datasets/"` strings |

Dates verified against the scikit-learn `whats_new` release notes and PyPI
upload timestamps. Where a feature predates every notebook in the repository by
years, saying so is the point.

`Pipeline` predates every notebook in this repository by six years, and
`ColumnTransformer` by about a year. `default_rng` is the one genuine
boundary case: NumPy 1.17 shipped in July 2019, and these notebooks are from
2019, so it was arguably not yet the obvious default.

The rest — `StratifiedKFold`, `OneHotEncoder`, `os.path.join`, PR-AUC,
group-aware splitting — has always existed. **This is a discipline gap, not a
tooling gap.** The single highest-value habit would have been to read the
documentation of the estimator I was already calling.

## 7.2 What genuinely did not exist in 2019

| Now | Then |
|---|---|
| Sentence-transformer embeddings | LDA over bag-of-words was the standard |
| Pretrained ViT/CNN backbones | HOG + SVM was competitive |
| CatBoost / LightGBM / modern GBDT | Random forests and tuned logistic regression |
| Optuna TPE, pruning | grid search, or genetic programming (TPOT) |
| `dowhy`, `rdrobust`, `rddensity` | hand-rolled RDD, as in this repository |
| DuckDB over Parquet | PostgreSQL + a zip file |
| `SimpleImputer`, `make_column_selector` | hand-rolled imputation loops |
| MLflow / W&B | a spreadsheet, or nothing |

Column 3 should be read as *"what I would do now"*, not *"what I did wrong in
2019"*. Choosing LDA in 2019 was a reasonable decision made with the available
options. The same reasoning does not apply in 2026.

## 7.3 The bug that shaped this retrospective

The most instructive thing found during the debugging pass is not any single
defect. It is that **three of the four worst bugs in the repository were silent**:

- a feature that could never be selected,
- a tree that predicted the exact inverse of what it learned,
- a cross-validation run that compared a model against itself,
- a label encoder that never ran.

None raise an exception. All four would have survived a code review, because
each reads as plausible logic that a reviewer has no reason to distrust. Three
of them were invisible until the *printed numbers* were read.

The one-line takeaway: **running the code and reading the output is a different
activity from running the code and checking the exit code.** Only the first one
finds this class of bug. That is now enforced by running all 19 notebooks in CI
(see Part 8).

---

# Part 8 — What is still not done

The external review is right about all of this, and none of it is finished.

| Item | Status | Priority |
|---|---|---|
| **CI** running tests, doctests, lint and notebook execution | absent | **High** — the single most valuable addition |
| **Exact dependency lock** (`uv.lock`) | absent — `requirements.txt` uses `>=` | High |
| **Checksums for downloaded datasets** | absent | Medium |
| **`notebooks/historical/` vs `notebooks/modernised/` split** | not done | Medium — see below |
| **`CHANGELOG.md`** | absent | Low |
| **Executed notebook outputs committed** | stripped, not re-executed | **High** — see below |

Two of these deserve elaboration.

**CI is the highest-value item.** The repository already has the thing CI needs
to be valuable — 19 notebooks that execute, 48 tests, 54 doctests. Wiring that
into GitHub Actions costs about twenty lines and makes the verification claim
continuous rather than a one-off. It also directly prevents the regression
class documented above, since the silent bugs were silent only because nobody
re-ran the notebooks after editing them.

**Notebook outputs should be committed.** A reviewer on GitHub reads the
rendered notebook, not the code. Outputs were stripped during the debugging pass
to remove 4.5 MB of `print()` spam from a runaway loop — correct at the time,
but the right end state is to re-execute cleanly and commit the results, so the
analysis is visible without running anything.

On the `historical/` vs `modernised/` split: I would **modify rather than adopt
this advice.** The review was written against a repository where six notebooks
did not run, so separating "broken history" from "modern work" seemed necessary.
Now that all 19 execute, they form a third category — *fixed historical work* —
which is neither. A `notebooks/modernised/` directory containing six rebuilt
projects, with the repaired originals left in place and cross-referenced, is the
right structure. The review's instinct (keep the history, make current ability
obvious) is correct; only the category names change.

**Do not rewrite all 19.** The review is right here too. Six representative
modernisations — RDD, credit approval, song/audio, bees transfer learning,
topic modelling, and the NumPy MLP — together cover statistics, tabular ML, raw
signals, vision, NLP and fundamentals. That is the demonstration.

---

# Part 9 — Recommendations the review did not make

These are my own additions, in priority order.

### 9.1 The repository name is a portfolio surface, and `small--projects` is a liability

`github.com/hwilner/small--projects` — the double dash is a rename artifact from
2019. It is the first thing a reviewer types, it looks like a typo, and it
carries no information about what the repository now contains. Renaming to
something like `ml-portfolio` or `applied-ml-work` is a thirty-second change
with a disproportionate effect on first impressions. Renaming preserves redirects
and does not break existing links.

### 9.2 There is no profile README

`github.com/hwilner` currently lands a reviewer on an undifferentiated list of
31 repositories — mostly forks, with two deleted, one public repo described as
"Internal tooling", and no indication of which three represent current ability.

A `hwilner/hwilner` profile README naming three projects, one line each, would
change the entire entry point. This is the highest-leverage change on this list
and it takes about twenty minutes.

### 9.3 `figure-pipeline` is public and labelled "internal tooling"

A public repo described as internal tooling is a mixed signal on a portfolio:
either it is worth showing, or it should be private. It is also the last
unresolved item from the original cleanup. **Recommendation: make it private.**
It is 33 KB, it is described as internal, and private is reversible.

### 9.4 Two repositories have unfinished publication hygiene

`matlab-neuroscience-coursework` still contains:

- **`contributors/ex7_Efrat_Sofer_3048515.m`** — a classmate's submission that
  arrived inside one of the archives. It is isolated and flagged, but it should
  be **deleted** along with `assignments/ex7_efrat_sofer_304855125.docx`
  unless there is explicit permission.
- **A student ID (`305571986`) in many filenames**, and graded assignment PDFs.
  For a public portfolio this may be more exposure than intended. `.gitignore`
  cannot help — the files are already in the history.

### 9.5 Pin the 2019 environment, not just the current one

The retrospective in this document argues that the 2019 work is worth preserving
*as 2019 work*. That claim is only credible if someone can reproduce it. Today
the historical notebooks run on 2026 libraries with API drift repaired by hand
(§Part 1 of the debugging record lists eight removed APIs).

An `environment-2019.yml` — scikit-learn 0.21, TensorFlow 1.x, pandas 0.25 —
would let a reader run the original code and see the original failures. That
turns the API-drift list from a claim in a markdown file into a demonstration.
It is also, quietly, a better artefact than the fixed version, because it shows
the delta.

### 9.6 Modernise one project enough to close the biggest skill gap

`SKILLS.md` names audio/DSP as the weakest area, and this retrospective agrees:
the one genuinely open gap. The song-genre project already has 13,129 tracks of
audio-derived features and the join logic. Computing the features — STFT,
spectrogram, MFCC — rather than accepting them pre-extracted, would close it in
one project and would move the repository from *applied statistics* into
*signal processing*, which is the single most valuable change available.

### 9.7 Numbers quoted in documentation will drift

This document and the README quote specific results (`0.824 vs 0.774`,
`0.830 held-out`, gradient errors of `4e-10`). Those are correct as of the
current commit and correct for the pinned seeds, but they will not survive a
re-run on different hardware or library versions without a note. Any
modernised notebook should write its headline result to a small
`results/<project>.json` that the documentation reads, rather than hard-coding
numbers in prose. Cheap now, painful later.

---

# Part 9b — The three columns, implemented

Everything above this point is prose describing what *would* be done. This
section is the thing itself.

## What was built

Six topics, three notebooks each — **18 notebooks**, every one of which
executes:

| Topic | What it covers |
|---|---|
| `01-causal-inference` | Regression discontinuity, from the 2019 error to bias-corrected inference |
| `02-tabular-fairness` | Credit approval, from a leaking pipeline to a fairness sweep |
| `03-audio-signal-processing` | A signal-processing chain written from scratch and verified |
| `04-nlp-topic-modelling` | Topic stability, coherence, and a representation comparison |
| `05-computer-vision` | HOG written out, verified, and compared on a synthetic corpus |
| `06-deep-learning` | A from-scratch MLP with verified gradients, through batch normalisation |

The three columns are the same as everywhere else in this document:

1. **2019, as written** — the original approach, faithfully reproduced so the
   next notebook has something to correct
2. **2019, with current judgement** — the same tools, the analysis done properly
3. **2026 tools** — what a modern stack adds, and where it does not help

Notebook source lives alongside as `.py` and is compiled by
`scripts/build_notebook.py`. Execution is enforced by
`scripts/run_notebooks.py`, which fails on any error. A notebook that does not
run is not part of the repository.

## The three columns, as notebooks

The analysis above is prose. These are the same three columns as code,
runnable, with every number below produced by a cell that executed.

Each topic is a directory under `notebooks/07-retrospective/`, with one
notebook per column. Source lives alongside as `.py` and is compiled by
`scripts/build_notebook.py`; execution is verified by
`scripts/run_notebooks.py`.

### 01-causal-inference — Regression discontinuity: bank debt recovery

Causal inference, the strongest project in the repository

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Regression Discontinuity: Bank Debt Recovery (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Regression Discontinuity: Bank Debt Recovery (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Regression Discontinuity: Bank Debt Recovery (2026 tools)

### 02-tabular-fairness — Credit approval and fairness

Tabular ML, and a lending decision examined for disparate impact

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Credit Approval (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Credit Approval (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Credit Approval (2026 tools)

### 03-audio-signal-processing — Song genre: computing the representation

Where the project moves from applied statistics to signal processing

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Song Genre from Audio Features (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Song Genre from Audio Features (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Song Genre from Audio (2026: computing the representation)

### 04-nlp-topic-modelling — Topic modelling on children's bestsellers

Bag-of-words versus embeddings, and the checks a topic model needs

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Hottest Topics in Children's Books (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Children's Books Topics (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Children's Books Topics (2026: embeddings vs bag-of-words)

### 05-computer-vision — Naive bees: HOG, PCA and SVM

A classical vision pipeline, reconstructed and verified

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Naive Bees: HOG + PCA + SVM (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Naive Bees: HOG + PCA + SVM (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Naive Bees: representation learning (2026)

### 06-deep-learning — Multilayer perceptron from scratch

The project called the 2019 retrospective's biggest missed opportunity

- `01-col-1-2019-written.ipynb` — **2019, as written**  
  1 — Multilayer Perceptron (2019, as written)
- `02-col-2-2019-judgement.ipynb` — **2019, with current judgement**  
  2 — Multilayer Perceptron (2019 judgement, same tools)
- `03-col-3-2026-tools.ipynb` — **2026 tools**  
  3 — Multilayer Perceptron (2026 framing)

**18 notebooks across 6 topics.** Every one executes, and `scripts/run_notebooks.py` fails the build if any does not.

## What the implementation changed about the analysis

Writing the notebooks was not a transcription exercise. Running them corrected
claims made earlier in this document, and the corrections are the substance of
the exercise.

**The causal inference result was a sign error.** Column 1 of the RD notebooks
reports −$641 for the effect of crossing the $1,000 threshold. Every local
window says +$403. The original 2019 analysis reported a confidently significant
result with the wrong sign, and the two lines of code that would have caught it
were already in the notebook. The corrected estimate is **+$249 (95% CI
[$156, $361])**, stable across bandwidths from $100 to $1,000. The lesson is not
about regression discontinuity; it is that a coefficient was never compared
against the descriptive statistic beside it.

**Batch-norm backpropagation was wrong in a way that trains.** The ReLU mask
belongs at the *top* of the backward pass, before the affine transform and the
batch-norm Jacobian. Applying it at the bottom is a plausible-looking change
that leaves the loss falling and every BN gradient wrong. The finite-difference
check found it; the training curve would not have. This is the fourth instance
of the same pattern in this repository, and the first one authored in 2026 —
which is the uncomfortable part.

**A positional-argument bug in the tempo estimator.** `onset_envelope(y, n_fft,
hop_length)` passed `n_fft` into the `sr` parameter, so the frame rate was
computed as `2048 / 512` instead of `22050 / 512`. Tempo search ran in the wrong
place and 120 BPM was reported as 60. No exception, plausible output, half the
table wrong by exactly a factor of two.

**Two claims in this document were false and are now corrected.** LDA coherence
was asserted to fall as the topic count grows; measured, it *rises*, because
umass averages over word pairs and small topics have fewer pairs to drag the
mean down. And the MLP improvement was expected to be a few points; measured,
Adam plus regularisation changed accuracy by less than a point, because the
2019 network was already near the ceiling of 8×8 digits and the regularisation
sweep selected nothing.

## What the honest limitations are

Three constraints shaped what these notebooks can claim, and none is fixable by
writing better code:

- **The bee photographs are gone.** Corrupted in the original repository and
  deleted during the debugging pass. The CV notebooks reconstruct and verify
  the *method* — HOG written out, checked against `skimage` — and run it on a
  synthetic corpus that is labelled as synthetic in every figure title.
- **There is no audio.** The echonest features arrive pre-extracted. The
  signal-processing chain is real and verified against `librosa` at >0.999
  correlation, but it runs on generated audio, and the result is about the
  pipeline rather than about FMA.
- **The topic-modelling corpus is a substitute.** The original NIPS abstracts
  had no stable source at the time, so the project ran on 603 NYT children's
  bestsellers. That corpus is too small to demonstrate the thing embeddings are
  better at, and the notebook says so instead of manufacturing a demonstration.

## Two gaps since closed, and what real data did to the claims

Both missing datasets turned out to be on Kaggle. Fetching them changed the
analysis rather than just filling it in, which is the fourth time in this
project that writing the check overturned the prose.

**Computer vision — the real BeeImage photographs (5,172 images).** HOG + SVM
reaches **0.8467** balanced accuracy against a **0.50** majority baseline, a
real result on a task that looks near-impossible at 64x64 greyscale. But
**augmentation hurts here, −0.29**, where it helped by +0.05 on the synthetic
corpus.

The reason is instructive. Real photographs already contain the variation the
augmentation invents — they were shot at different angles, distances and
lighting. The synthetic corpus came from a single template, so augmentation was
adding genuine invariance that the generator happened not to vary. **The
synthetic result was true about the synthetic corpus and wrong about bees.**
That is the strongest argument in this repository for labelling synthetic data
as synthetic rather than quietly dropping it.

The recovered dataset annotates subspecies and health, **not genus**, so the 2019
task is not reproducible from it. The notebook runs health classification and
states plainly that it is a different task.

**NLP — the real NIPS abstracts (6,359 documents).** NMF recovers topics that
are recognisable research areas rather than stock phrases, and the era × topic
association is significant at **p = 6.6e-09**: a single k-topic model over the
corpus is partly a model of when a paper was written.

Two claims had to be withdrawn on contact with the real data:

- The dump is advertised as 1987–2019, and `year` genuinely spans that. But
  **abstract coverage starts in 2007** — 4 papers before then carry one, against
  3,124 that have full text. An era analysis reaching to 1987 compares a
  119-word vocabulary against a 16,000-word one. The notebooks bin on 2007+ and
  assert that every bucket clears 50 documents.
- The 0.4 "dominant topic" threshold from column 2 is an **LDA** threshold, where
  the document-topic weights form a probability distribution. NMF factors are
  non-negative and unbounded, so the same threshold reads 100% unassigned. The
  notebook reports that as a category error rather than quietly dropping it.

**The audio gap remains open.** No audio dataset is in the repository, so the
signal-processing chain is verified against `librosa` at >0.999 correlation on a
synthetic signal and the classifier comparison runs on generated audio. An FMA
copy exists on Kaggle (`aaronyim/fma-small`) but is 8 GB of audio for a task
needing a few hundred tracks; it is documented and unfetched rather than
half-fetched.

The honest move is the same one as before: state the gap, verify what can be
verified, and never present a synthetic result as a real one. Two gaps closed,
one open, and two of my own claims retracted.

## The numbers, as executed

Every figure below was produced by a cell that ran. Nothing here is quoted from
the prose above.

| Topic | Result |
|---|---|
| Causal inference | Effect **+$249**, 95% CI [$156, $361], bias-corrected with an MSE-optimal bandwidth of $760. McCrary p = 0.54, no slope change. |
| Credit, column 2 | CV balanced accuracy 0.8771 vs **0.8272** held out — CV optimistic by 0.05. Calibration error 6.7%. Approval-rate gap by marital status **p = 0.0197**. |
| Credit, column 3 | Nested-CV estimate, isotonic calibration, and a selection-rate ratio below 0.80 across most of the threshold range. |
| Audio | MFCCs correlate with `librosa` at **>0.999**. Tempo recovered within **0.9 BPM** across 80–180 BPM. Handcrafted 68 features vs pooled log-mel 128: **0.785 vs 0.987** on the synthetic corpus. |
| NLP | Topic count chosen by held-out perplexity: **k = 3**. Stability 0.344 at k=3 against **0.074 at k=14**. 108 of 420 documents unassigned. |
| Vision | HOG verified against `skimage` (lengths identical, soft vs hard binning documented). HOG over raw pixels **+0.05**; **PCA over no-PCA: no gain**. |
| Vision, **real data** | Kaggle BeeImage, 5,172 photographs. HOG + SVM reaches **0.8467** balanced accuracy against a **0.50** majority baseline, ROC-AUC 0.9232 — and augmentation **hurts** (−0.29), reversing the synthetic result. |
| NLP, **real data** | Kaggle NIPS dump, 6,359 abstracts. NMF recovers recognisable subfields. Era is confounded with topic, **p = 6.6e-09**. Vocabulary Jaccard against 2007–09 decays 0.476 → 0.321 by 2016–19. |
| MLP | Gradients match finite differences to **3.6e-10**, dropout path to **1.6e-10**, batch-norm path to **2.5e-10**. Zeros init: **0.1000** accuracy, 100% dead ReLUs. |

Two of these deserve emphasis because they contradict the plan rather than
confirming it.

**LDA coherence rises as the topic count grows.** The expectation was that more
topics means less structure. Measured, umass coherence goes from 2.75 at k=3 to
4.16 at k=14, because the metric averages over word *pairs* and a small topic
has fewer pairs to drag the mean down. Coherence alone cannot select k. Stability
can, and stability is what condemns k=14.

**Batch norm made the model worse, correctly.** 0.9367 against the plain
network's 0.9778, with gradients verified correct to 2.5e-10 including the
running statistics used at eval time. That is not an implementation failure; it
is batch norm doing what it does on a two-layer MLP trained on 1,437 images.
The number is reported because it is the result, not because it is flattering.

## The pattern across all six

Six projects, four silent bugs, and every one had the same signature:

| Bug | How it presents | What caught it |
|---|---|---|
| RD coefficient with the wrong sign | p = 1.3e-17, highly significant | comparing it to the local means already computed |
| Batch-norm mask ordering | loss falls, accuracy looks plausible | finite differences |
| Positional args into the wrong parameter | returns a plausible float, half the table wrong | noticing every value was wrong by exactly 2x |
| Gradient check comparing unchecked zeros | an "error" of 0.52 that was an artefact | reading the failure instead of tuning around it |

None of them raises. None of them is visible in a training curve. All four are
visible in a number, and three of the four were found by writing a check rather
than by reading the code.

That is the retrospective's actual finding, and it is now demonstrated rather
than asserted. The 2019 versions were not wrong because the tools were missing.
They were wrong because nothing was ever compared against anything.

# Part 10 — The three-sentence version

**2019:** broad, curious, statistically better than average for the era, and
structurally undisciplined — most defects were available-to-fix in 2019 and
simply not fixed.

**2019 with current knowledge:** the same projects would score higher almost
entirely through discipline rather than libraries. `Pipeline`,
`StratifiedKFold`, `OneHotEncoder`, an honest baseline, an untouched test set,
a gradient check. None of that needed a library that did not exist.

**2026:** the projects that survive contact with a modern reviewer are the ones
that changed *category* — causal inference with proper refutation, audio
features computed rather than imported, transfer learning measured against
handcrafted features, embeddings measured against bag-of-words. Rewriting
everything would waste the history; rebuilding six would demonstrate the
evolution better than any amount of updating.

**And the meta-lesson, which is the part actually worth keeping:** the three
worst bugs in this repository were silent. They produced no error, survived
inspection, and were only visible in the numbers they printed. The habit that
would have caught all of them in 2019 is not a library, a framework, or a
technique. It is: *run it, and read what it says.*
