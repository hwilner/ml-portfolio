# Bugs found and fixed

Every notebook in this repository was executed top to bottom after the
restructure, and every failure was fixed rather than documented around. This is
the record of what was wrong, what it cost, and how each one was verified.

**Result: 19 notebooks, 19 passing, 0 failing.**

The packages are verified separately:

```
48 behavioural tests    54 doctests    2 fetch-script doctests    0 failures
```

---

## The ones that mattered most

### 1. A decision tree that could never use the first feature

```python
if ent[col][ind] < min_avg_ent and col:      # <-- the bug
```

`col` is a column index. Column `0` is falsy in Python, so `and col` was False
every time the first feature was the best candidate — **feature 0 could never be
selected, for any dataset, ever.** No exception, no warning. The tree simply
trained worse than it should have, and a reader had no way to tell.

This is the worst class of bug in the set: no error message, no traceback,
just quietly worse results.

*Fix:* the clause is gone; every feature is a candidate. Verified by
`test_feature_zero_is_usable` in the test suite, and by a regression test in
the notebook itself.

### 2. A decision tree that learned the right split and predicted its opposite

Found only by executing the notebook and reading the numbers. The build step
passed samples **above** the threshold to the left child, while `predict()`
routed samples **below** the threshold to the left child.

The cells ran. The split was correct. The tree scored **0.0 accuracy** on
perfectly separable data — worse than random, which is the signature of a
systematic inversion rather than a weak model.

*Fix:* the left child is now the samples satisfying the routing condition.
Verified by a `min_samples_leaf` table that now shows the textbook
bias-variance curve:

| `min_samples_leaf` | nodes | depth | train acc | test acc |
|---:|---:|---:|---:|---:|
| 1 | 81 | 14 | 1.0000 | 0.8800 |
| 5 | 47 | 11 | 0.9500 | 0.8600 |
| 20 | 23 | 5 | 0.8867 | 0.8800 |
| 50 | 11 | 3 | 0.8767 | **0.9100** |

### 3. Backpropagation that was never going to work

The original `backprop` reversed the layer list (`weigths_layer[::-1]`) while
`feed_forw` walked it forward, produced a hidden-layer update
`np.sum(y_hat.reshape(-1,1) @ w.reshape(1,-1), axis=0)` whose shape could not
match its target, called an undefined `argmax`, and had `fit()` return `None`.

No patch fixes that. The cause was structural: weights stored as flat 1-D
arrays with the bias prepended to the input on every layer, so a weight vector
had to match its own *input* dimension.

*Fix:* rewritten with explicit `(n_in, n_out)` matrices per layer and separate
bias vectors. Forward is `x @ W + b`; the gradient is `delta @ x.T`. Neither
the ordering bug nor the shape bug can occur.

*Verified* by a finite-difference gradient check:

```
param    max abs difference   verdict
W1                3.994e-10   MATCH
b1                1.270e-10   MATCH
W2                3.280e-10   MATCH
b2                2.266e-10   MATCH

training accuracy: 0.9979
test accuracy:     0.9750
```

A hand-derived gradient that has not been checked against numerical
differencing is not a result, it is a guess.

### 4. A cross-validation comparison of a model against itself

```python
tree_score  = cross_val_score(tree,  ...)
logit_score = cross_val_score(tree,  ...)   # <-- should be `logreg`
```

Logistic regression was never scored. The printed "Decision Tree vs Logistic
Regression" comparison was the decision tree measured against itself, and the
notebook's conclusion rested on it.

*Fix:* plus `StratifiedKFold` (the class ratio drifts under plain `KFold`), and
a paired t-test so the difference is reported with uncertainty rather than as
two bare means.

*Verified* — and the true result is more interesting than the bug was:

```
Decision Tree       mean=0.7736  sd=0.0384
Logistic Regression mean=0.8242  sd=0.0200
difference (logreg - tree): +0.0505
paired t-test across folds: t=4.41, p=0.002
```

### 5. Test-set leakage hidden behind a computed-and-discarded variable

```python
rescaledX = scaler.fit_transform(X)       # fitted on ALL of X, test included
grid_model_result = grid_model.fit(X, y)  # ...and then never used rescaledX
```

The scaler was refitted on the full dataset, leaking test statistics, and the
scaled array it produced was then thrown away — the search ran on raw data.
The reported `best_score` was optimistic.

*Fix:* a `Pipeline`, so the scaler is fitted inside each CV fold on that fold's
training portion only.

*Verified* — and honest reporting makes the problem visible:

```
Best: 0.852174 using {'logreg__max_iter': 100, 'logreg__tol': 0.01}
fold-to-fold std at the best config: 0.1372
tuned pipeline on the held-out test split: 0.8296 (+/- 0.0613)
```

Cross-validation claimed 0.852; the model actually scores **0.830** on held-out
data, with a fold-to-fold standard deviation of 0.137. The tuning gain was
mostly noise. The original bug hid that.

### 6. Four JPEG files that were not images

The four `bee_*.jpg` files committed to the original repository were corrupt.
Every byte `>= 0x80` had been replaced with a UTF-8 replacement character
(`EF BF BD`) — the files were uploaded through a text pipeline rather than as
binary. `PIL.Image.open` raised `UnidentifiedImageError` on all four.

The corruption is not reversible. The header `FF D8 FF E0` became four
identical replacement sequences, so `D8` and `E0` are indistinguishable from
`FF` and cannot be recovered; the entropy-coded scan data is destroyed too.
Reconstructing the standard JFIF header and restoring `0xFF` everywhere was
tried and still failed to decode.

*Fix:* the corrupt files are deleted. Both CV notebooks now generate a
clearly-labelled synthetic stand-in when real images are absent, and say so on
every figure, because a perfect score on drawn shapes is not evidence.

### 7. `sel.patience` — an undefined name in a notebook that could never run

`patience_left = sel.patience + 1` raised `NameError` before the early-stopping
loop could begin. The notebook was also titled "mini batch GD" while computing
a full-batch update, and applied L2 with double the weight of L1 at equal `r`.

*Fix:* rewritten with `batch_size` as a real parameter, a correct penalty
gradient, and early stopping that actually halts.

### 8. An inverse transform that inverted nothing

`un_norm` read `X *= (max + min) + min`. The inverse of `(x - min) / (max - min)`
is `(x * (max - min)) + min`. As written it did not undo the forward transform.

Underneath that sat a worse problem: the design matrix was rebuilt from the
*data* inside `predict`, so the weights had been fitted against one basis and
applied to a different one. Predictions on new points were meaningless.

*Fix:* the basis is fixed at fit time. The notebook now demonstrates the
consequence directly — fitting on `x < 1.5` and predicting on `[1.5, 3)`, a
test the original design could not have passed.

### 9. Renaming a parameter silently changed what a call meant

`load_data(..., size=2000)` meant *2000 samples*. After the loader was
documented and given a proper signature, `size` became the *image side length*
and the sample count became `limit`. The unchanged call then asked for
2000×2000-pixel images: 48 MB each, 120 of them, and the kernel was
OOM-killed with exit 137 before printing anything.

*Fix:* both arguments passed by name. Worth remembering — renaming a parameter
does not break the call that uses it correctly, it breaks the one that used it
by position, and it fails far from the edit.

---

## API removals

Every one of these is a hard failure on a current install, not a deprecation
warning.

| Removed | Replacement | Where |
|---|---|---|
| `get_feature_names()` | `get_feature_names_out()` | LDA notebook, ×2 |
| `ENGLISH_STOP_WORDS` import | `stop_words="english"` | LDA notebook |
| `rgb2grey` | `rgb2gray` | Naive Bees HOG |
| `tf.set_random_seed` | `tf.random.set_seed` | ASL CNN |
| `keras.utils.np_utils` | `np.eye(n)[y]` | ASL CNN |
| `DataFrame.as_matrix()` | `.to_numpy()` | Credit card |
| `sklearn.externals.six` | `io.StringIO` | Tree visualisation |
| TPOT 0.x `generations` / `population_size` / `verbosity` / `config_dict` | `search_space` / `max_time_mins` / `verbose` | TPOT notebook |

---

## Silent no-ops (the ones that never raise)

These are the reason the suite was run rather than the notebooks merely
re-read. None of them produces an error; each just quietly does the wrong
thing.

| Bug | Effect |
|---|---|
| `cc_apps[col].dtype == 'object'` | pandas 3 gives text columns a `str` dtype, so the label encoder **never ran** and raw strings reached `MinMaxScaler` |
| `DataFrame.mean()` on a mixed frame | "Cannot perform reduction 'mean' with string dtype" — the numeric-only selection was missing |
| `corr()` on a frame with a string column | "could not convert string to float: 'Hip-Hop'" |
| `and col` in the split guard | feature 0 unreachable (bug 1) |
| `rescaledX` computed then unused | test-set leakage (bug 5) |
| `tol` documented but never read | early stopping fired on floating-point noise, or never fired at all |
| `error()` returned a count of *correct* predictions | the training loop's stopping condition was true forever |
| both perceptron update branches identical | the rule could not learn; the sign has to come from the branch |

---

## Runtime failures

| Bug | Cause |
|---|---|
| `SyntaxError: incomplete input` | a cell opened `"""` and closed `'''` |
| `StdinNotImplementedError` | `input()` called during non-interactive execution |
| `IndexError: invalid index to scalar variable` | 1-D input iterated as scalars; also left/right children swapped (bug 2) |
| `ValueError: non-broadcastable output operand` | `xb @ weights` is `(n,)`, so `(n,) - (n, 1)` silently became `(n, n)` |
| `TypeError: only 0-dimensional arrays can be converted` | `float()` on a length-1 array, rejected in NumPy 2 |
| `ufunc 'bitwise_and' not supported` | `radius < a & radius > b` — `&` binds tighter than `<` |
| `IndexError: index 3 out of bounds for axis 2 with size 3` | `enumerate(..., start=1)` used as a channel index |
| `KeyError: 'year'` | the column is `Year` |
| `ufunc 'floor_divide'` on the decision tree | `data_n` referenced instead of `data`; `predict` read a never-populated `self.labels` |
| `AttributeError: 'DataFrame' has no attribute 'as_matrix'` | pandas 1.0 removal |
| `AttributeError: 'NoneType' object has no attribute 'generate'` | `search_space=None` is not a valid TPOT 1.x value |
| `exit 137` | 2000×2000 images (bug 9) |

---

## Broken datasets

| Dataset | Status |
|---|---|
| `papers.csv.gz.part_aa`…`part_ah` | A 110 MB multi-part download that was **never reassembled**. Deleted; the LDA notebook falls back to a labelled synthetic corpus, and `fetch_data.py --only lda` explains where to get the real thing |
| `bee_*.jpg` ×4 | Irrecoverably corrupted (bug 6) |
| ASL `train.zip` | The Udacity S3 bucket no longer exists. The loader is shipped and tested; the notebook generates labelled synthetic letters |

---

## Inferred errors — bugs found by reading the code

A small number were caught by inspection rather than execution, and fixed on
the same pass:

* an **infinite loop** in the perceptron: the error count was measured once
  before the loop and never refreshed, and it counted *correct* predictions, so
  the stopping condition was permanently true
* an **unbounded `while`** in momentum descent, with no iteration cap — a bad
  starting point hangs the notebook
* **aliasing**: `wt = w` followed by `wt[-1] = b` overwrote a real weight
  through the alias on every pass
* **swapped argument order** between a perceptron's `y_pred(w, x)` definition
  and all of its call sites

---

## What this says about the code

Most of these are not exotic mistakes. The dominant pattern is **API drift** —
code written against a library version and never run again — which is exactly
what you would expect from notebooks that were explorations rather than
projects.

The second pattern is quieter and more interesting: bugs that *silently
produce wrong numbers*. The inverted decision tree, the disabled first feature,
the model-against-itself cross-validation, the discarded scaling, the
label encoder that never ran. None of those raise. All of them would have
survived a code review, because reading the code shows plausible logic that a
reader has no reason to distrust.

Which is the argument for running things, and for checking the output rather
than just the exit code. Two of the worst bugs here — the inverted tree and the
useless cross-validation — produced no error at all. They were only visible
because the tables and the printed numbers were read.
