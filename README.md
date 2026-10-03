# Machine Learning & Data Science Projects

A collection of applied machine-learning, statistics and data-analysis projects,
built as a record of practical work: statistical reasoning, algorithm
implementation from first principles, and end-to-end modelling pipelines.

Every project is a Jupyter notebook. The algorithms that were worth making
reusable now also exist as a tested, installable Python package under
[`src/ml_from_scratch/`](src/ml_from_scratch/).

---

## Table of contents

- [Start here](#start-here)
- [Repository layout](#repository-layout)
- [Projects](#projects)
- [Skills demonstrated](#skills-demonstrated)
- [The `ml_from_scratch` package](#the-ml_from_scratch-package)
- [Running the code](#running-the-code)
- [Datasets](#datasets)
- [Verification](#verification)
- [Bugs found and fixed](#bugs-found-and-fixed)
- [Retrospective: 2019 → 2026](#retrospective-2019--2026)
  - [The three columns, as runnable notebooks](#the-three-columns-as-runnable-notebooks)
- [Contributing](#contributing)

---

## Start here

**If you only have two minutes**, read these in order:

1. **Regression discontinuity on bank debt recovery**
   ([`02-statistics/`](notebooks/02-statistics/)) — the strongest single
   project here. It is a *causal inference* problem, not just a prediction
   task: it asks whether a bank's collection policy changes how much money it
   actually recovers, and answers with a regression discontinuity design backed
   by hypothesis tests.
2. **The `ml_from_scratch` package** ([`src/ml_from_scratch/`](src/ml_from_scratch/))
   — perceptron, linear regression with regularisation, and a CART decision
   tree, all written against NumPy alone, all covered by 48 behavioural tests.
   This is the part that is engineered rather than exploratory.

For a full competency-by-competency breakdown, see **[SKILLS.md](SKILLS.md)**.
For the debugging record, see **[BUGFIXES.md](BUGFIXES.md)**.

---

## Repository layout

```
.
├── README.md                  ← you are here
├── SKILLS.md                  ← competency matrix, per project and per skill
├── BUGFIXES.md                ← what was broken, and how it was verified
├── RETROSPECTIVE.md           ← 2019 as written / 2019 redone / 2026
├── notebooks/07-retrospective/ ← 18 notebooks: 6 topics x 3 columns
├── notebooks/
│   ├── 01-nlp/                ← topic modelling, phonetic string matching
│   ├── 02-statistics/         ← causal inference, hypothesis testing
│   ├── 03-machine-learning/   ← classical ML pipelines, AutoML
│   ├── 04-computer-vision/    ← image features, CNN
│   ├── 05-algorithms-from-scratch/  ← exploratory algorithm notebooks
│   └── 06-sql/                ← relational analysis
├── src/ml_from_scratch/       ← the tested, reusable implementations
├── tests/                     ← 48 behavioural tests
├── data/                      ← small datasets, committed
├── scripts/fetch_data.py      ← downloads the datasets too large for git
├── requirements.txt
└── pyproject.toml
```

Numbered prefixes keep folders in a sensible reading order on GitHub and make
it obvious where a project sits in the progression from exploratory to
engineered.

---

## Projects

### Statistics & causal inference

| Project | What it does | Why it matters |
|---|---|---|
| [Regression discontinuity: bank debt recovery](notebooks/02-statistics/01-regression-discontinuity-bank-recovery.ipynb) | Tests whether a bank's escalating collection strategy, triggered at debt thresholds ($1k/$2k/$3k/$5k), causally changes the amount actually recovered. | A genuine **causal inference** design. Runs balance tests (Kruskal–Wallis, chi-square) to check the threshold is plausibly random, then estimates the treatment effect with OLS and an indicator, at two different bandwidths. |

### Algorithms from first principles

Every algorithm in [`src/ml_from_scratch/`](src/ml_from_scratch/) was
originally worked out in a notebook and then rewritten as reviewed,
documented, tested code. The notebooks are kept as the record of the process.

| Module | Algorithm | Notebook it came from |
|---|---|---|
| [`perceptron.py`](src/ml_from_scratch/perceptron.py) | Perceptron learning rule, with epoch bounding and early stopping | `05-algorithms-from-scratch/linear-models/05-`, `06-` |
| [`linear_regression.py`](src/ml_from_scratch/linear_regression.py) | Min-max scaling, gradient descent, ridge/L1/L2, early stopping, mini-batching | `05-algorithms-from-scratch/linear-models/01-`, `02-`, `03-` |
| [`decision_tree.py`](src/ml_from_scratch/decision_tree.py) | Shannon entropy, information gain, greedy CART splitting, min-leaf-size regularisation | `05-algorithms-from-scratch/tree-models/01-` |

### Machine learning pipelines

| Project | Problem | Highlights |
|---|---|---|
| [Classify song genres from audio data](notebooks/03-machine-learning/01-classify-song-genres-pca.ipynb) | Rock vs hip-hop from 8 pre-extracted audio features | Correlation filtering, PCA with scree and cumulative-variance plots, decision tree vs logistic regression, class rebalancing, K-fold cross-validation |
| [Predicting credit card approvals](notebooks/03-machine-learning/02-credit-card-approvals.ipynb) | Binary classification on bank applications | Missing-value analysis, mean and mode imputation chosen per column type, label encoding, min-max scaling, `GridSearchCV` tuning |
| [Automatic model selection with TPOT](notebooks/03-machine-learning/03-tpot-blood-donations-automl.ipynb) | Predict blood-donation likelihood from RFM data | Automated ML via genetic programming, AUC evaluation, log transform justified by variance inspection, stratified splitting |

### Computer vision

| Project | Problem | Highlights |
|---|---|---|
| [Naive Bees: image manipulation](notebooks/04-computer-vision/01-naive-bees-image-manipulation.ipynb) | Image processing fundamentals | RGB channel separation, kernel density estimates per channel, greyscale conversion, contrast stretching, batch transformation pipelines |
| [Naive Bees: HOG + PCA + SVM](notebooks/04-computer-vision/02-naive-bees-hog-pca-svm.ipynb) | Bee species from images | Histogram of Oriented Gradients descriptors, PCA to 500 components, SVM with probability calibration, ROC curve and AUC |
| [ASL recognition with a CNN](notebooks/04-computer-vision/03-asl-recognition-cnn.ipynb) | Classify ASL letters from photographs | Two conv/pool blocks then a dense softmax, one-hot labels, misclassification analysis |

### NLP

| Project | Problem | Highlights |
|---|---|---|
| [The hottest topics in machine learning](notebooks/01-nlp/02-hottest-topics-lda.ipynb) | Discover research themes in NIPS titles | Text cleaning, stop-word removal, bag-of-words, **Latent Dirichlet Allocation** for unsupervised topic discovery, word clouds |
| [Name game: gender prediction via NYSIIS](notebooks/01-nlp/01-name-gender-prediction-nysiis.ipynb) | Predict author gender from first names | Phonetic encoding with the New York State Intelligence System algorithm, fuzzy matching against SSA baby-name data, frequency analysis. **Method is weak on its own terms** — the notebook's own caveat explains why |

### SQL

| Project | Problem | Highlights |
|---|---|---|
| [International debt statistics](notebooks/06-sql/01-international-debt-statistics.ipynb) | Analyse World Bank debt data | Aggregation, `GROUP BY`, correlated subqueries, ordering by aggregates, scale-aware number formatting |

---

## Skills demonstrated

An honest map of what this repository shows. Full detail, including where the
gaps are, is in **[SKILLS.md](SKILLS.md)**.

| Skill | Level | Strongest evidence |
|---|---|---|
| **Statistical inference** | Strong | Regression discontinuity design, Kruskal–Wallis, chi-square tests of balance, OLS with interaction terms |
| **Causal inference** | Strong | The RDD project; bandwidth sensitivity analysis |
| **Algorithm implementation** | Strong | 48 passing tests over perceptron, GD regression with regularisation, and CART from scratch |
| **Data science / wrangling** | Strong | Missing-value strategy selection, variance-driven log transforms, class rebalancing, correlation-based feature selection |
| **Classical ML** | Strong | PCA, decision trees, logistic regression, SVM, cross-validation, hyperparameter search |
| **Unsupervised learning** | Solid | LDA topic modelling with `min_df`/`max_df` pruning, PCA |
| **Computer vision** | Solid | HOG descriptors, CNN architecture design, image feature engineering |
| **Image signal processing** | Developing | Greyscale, gradient filters, block normalisation, contrast operations |
| **Neural networks** | Solid | CNN design and training; backpropagation from scratch, **verified against finite differences to 1e-10** |
| **NLP** | Working | LDA, bag-of-words, stop-word pruning, NYSIIS phonetic encoding |
| **SQL** | Solid | Aggregation, subqueries, grouping |
| **Audio / DSP** | **Weak** | Audio *descriptors* are used pre-extracted; no raw-signal filtering, FFT or spectral work |
| **MLOps / deployment** | **Not here** | No serving, tracking or pipeline orchestration in this repo |

**Read that table as the honest summary.** The two bold entries are where a
reviewer would push back, and both are accurate. See
[SKILLS.md](SKILLS.md#gap-analysis) for what would close them.

---

## The `ml_from_scratch` package

The notebooks are exploratory by nature: exploratory, iterative, and often
mid-thought. The package is the same ideas written to be relied on.

```python
from ml_from_scratch import Perceptron, LinearRegression, DecisionTreeClassifier

# A perceptron over 21 binary features, no sklearn involved.
clf = Perceptron(n_epochs=30, learning_rate=0.1, random_state=0)
clf.fit(X, y)
print(clf.score(X, y), clf.errors_[-1])

# Ridge-regularised regression with early stopping on a validation split.
reg = LinearRegression(penalty="l2", alpha=0.1, early_stopping=True, patience=20)
reg.fit(X, y)
print(reg.coef_, reg.intercept_, reg.score(X, y))

# A CART tree that can split on feature 0 — and stops on minimum leaf size.
tree = DecisionTreeClassifier(min_samples_leaf=20, max_depth=4)
tree.fit(X, y)
print(tree.score(X, y), tree.max_depth_, tree.n_nodes_)
```

Every public function and class carries a
[Google-style docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
with `Args:`, `Returns:`, `Raises:` and a runnable `Example:`.

### Quality bar

```
48 behavioural tests    54 doctests    0 failures
```

The tests do not just check that functions return values. They check that the
algorithms **learn the right thing**:

- the perceptron drives its error count to zero on separable data
- linear regression recovers known coefficients and is invariant to feature
  scaling (which is the entire reason the module scales)
- L1 regularisation shrinks a pure-noise feature harder than OLS does
- the decision tree's leaves never fall below `min_samples_leaf`
- `min_samples_leaf` genuinely reduces node count
- the tree **can** split on feature 0 — the specific bug that crippled the
  original notebook (see below)

Several tests exist purely as regression tests for bugs found in the original
notebooks. They are named after the bug, so the reason they exist stays
legible.

---

## Retrospective: 2019 → 2026

### The three columns, as runnable notebooks

`notebooks/07-retrospective/` contains **18 notebooks — three per topic, one for
each column of the analysis.** Every one executes:

| Topic | 2019 as written | 2019 with judgement | 2026 tools |
|---|---|---|---|
| `01-causal-inference` | the −$641 sign error | McCrary, placebo cut-offs, CV bandwidth | bias-corrected, MSE-optimal bandwidth |
| `02-tabular-fairness` | leaking pipeline | `Pipeline`, `OneHotEncoder`, cost-based threshold | CatBoost, Optuna, nested CV, fairness sweep |
| `03-audio-signal-processing` | pre-extracted features | `Pipeline`, no resampling, McNemar | STFT → mel → MFCC written from scratch |
| `04-nlp-topic-modelling` | k=14 by eye | stability, coherence, held-out k | embeddings, UMAP, c-TF-IDF labels |
| `05-computer-vision` | HOG + PCA + SVM | Pipeline, hyperparameter search, baselines | augmentation, TTA, PCA spectrum |

Two notebooks go past the three columns, because the missing datasets turned out
to be recoverable and the real data changed the conclusions:

- `04-nlp-topic-modelling/04-real-data-nips-kaggle.ipynb` — **6,359 real NIPS
  abstracts.** NMF recovers recognisable subfields; era is confounded with topic
  at p = 6.6e-09. Also corrects the dataset's advertised 1987–2019 span: abstract
  coverage starts in 2007.
- `05-computer-vision/04-real-data-kaggle-bees.ipynb` — **5,172 real bee
  photographs.** HOG + SVM hits 0.8467 balanced accuracy against a 0.50
  majority baseline. Augmentation *hurts* by 0.29, reversing the synthetic
  result.

Fetch those two datasets with the commands in `data/nlp/README.md` and
`data/computer-vision/README.md`; the third-party data is not committed.
| `06-deep-learning` | backprop was a stub | Adam, dropout, **gradient check** | batch norm, verified gradients |

Build and verify them with:

```bash
python scripts/build_notebook.py                     # compile .py -> .ipynb
python scripts/run_notebooks.py 07-retrospective     # execute; fails on error
```

[`RETROSPECTIVE.md`](RETROSPECTIVE.md) reviews every project in three columns:
**as written in 2019**, **how 2019 would look redone with 2026 judgement**, and
**how it would be built in 2026**.

The distinction that matters is between *"I didn't know better"* and *"the
tool didn't exist yet."* Most of the defects in this repository are the first
kind — `Pipeline` predates every notebook here by six years, and
`ColumnTransformer` by about a year. That is a discipline gap, not a tooling
gap, and it is the more fixable of the two.

It also documents what is still undone (CI, a dependency lock, dataset
checksums, committed notebook outputs) and makes recommendations the review
that fed into it did not — the repository name, the missing profile README, and
pinning a 2019 environment so the historical code stays runnable.

## Running the code

```bash
# 1. Create an environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the full notebook stack
pip install -r requirements.txt

# 3. Or just the tested package, which only needs NumPy
pip install -e .

# 4. Fetch the datasets that are too large for git (optional, ~160 MB)
python scripts/fetch_data.py

# 5. Run the tests
pytest tests/ -v
```

Notebooks are grouped by area, so you can install only what you need — the
statistics projects need roughly `numpy pandas scipy statsmodels
matplotlib`, and nothing else.

---

## Datasets

Small datasets are committed under [`data/`](data/). Anything above ~1 MB is
fetched on demand by [`scripts/fetch_data.py`](scripts/fetch_data.py), which
keeps the repository clone fast. The largest single dataset removed this way was a **110 MB** broken
multi-part download that was never reassembled. `fetch_data.py` now carries
only URLs verified to return HTTP 200, and gives manual instructions for the
three datasets that have no usable raw mirror.

See [`data/README.md`](data/README.md) for the full manifest, provenance and
licences.

---

## Verification

Every notebook is executed top to bottom as part of the project's checks. The
current state:

```
19 notebooks    19 passing    0 failing

src/ml_from_scratch    48 tests + 54 doctests, 0 failures
scripts/fetch_data.py  2 doctests, 0 failures
```

Notebooks that need data they cannot download generate a **clearly-labelled
synthetic stand-in** and print a warning, so a fresh clone still runs end to
end. Where that happens, every figure is titled `SYNTHETIC DATA` and any
accuracy printed alongside describes synthetic shapes rather than photographs.

## Bugs found and fixed

The restructure was followed by running everything, and running things turned
up bugs that reading the code did not. The full record is in
**[BUGFIXES.md](BUGFIXES.md)**; the three worth knowing about:

**A decision tree that could never use its first feature.** The split guard
read `if ent < min_avg_ent and col:`. Since `0` is falsy in Python, column 0
could never be selected — for any dataset, with no error message. The tree
simply trained worse than it should have.

**A decision tree that predicted the exact opposite of what it learned.** The
build step passed samples above the threshold to the left child while
`predict()` routed samples below the threshold there. Every cell ran; the split
was correct; the model scored **0.0** on perfectly separable data.

**A cross-validation comparison of a model against itself.**
`logit_score = cross_val_score(tree, ...)` scored the decision tree twice, so
the notebook's "Decision Tree vs Logistic Regression" conclusion rested on
nothing. Fixed, the real result is logreg **0.824 vs tree 0.774**, p = 0.002.

Most of the rest is API drift: `get_feature_names()`,
`ENGLISH_STOP_WORDS`, `rgb2grey`, `tf.set_random_seed`, `np_utils`,
`as_matrix()`, `sklearn.externals.six`, and the whole TPOT 0.x search
interface were all removed after these notebooks were written.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: the package must stay NumPy-only
and keep 100% doctest and test coverage; notebooks should be runnable or
explicitly marked as not.

## Licence

[MIT](LICENSE).
