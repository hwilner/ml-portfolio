# Algorithms from scratch

Exploratory notebooks where learning algorithms were derived by hand, before
any library was allowed to do the work.

**All nine now execute end to end.** Most needed real fixes rather than
cosmetic ones — including a perceptron whose two update branches ran the same
line, and a decision tree that could never use its first feature. The full
record is in [`BUGFIXES.md`](../../BUGFIXES.md); the per-notebook detail is
below, kept because *how* a bug was found is the interesting part.

The algorithms were also rewritten as a tested package — see
[`src/ml_from_scratch/`](../../../src/ml_from_scratch/).

| Working implementation | Replaces |
|---|---|
| [`perceptron.py`](../../../src/ml_from_scratch/perceptron.py) | `linear-models/05-`, `06-` |
| [`linear_regression.py`](../../../src/ml_from_scratch/linear_regression.py) | `linear-models/01-`, `02-`, `03-` |
| [`decision_tree.py`](../../../src/ml_from_scratch/decision_tree.py) | `tree-models/01-` |

---

## Status

| Notebook | Was | Now |
|---|:--:|---|
| `linear-models/01-linear-regression-gradient-descent.ipynb` | ⚠️ | ✅ Syntax error from mismatched quote delimiters, plus a blocking `input()` call. Reference for the package. |
| `linear-models/05-perceptron-from-scratch.ipynb` | ✅ | ✅ Clean. Reference for the package. |
| `linear-models/04-momentum-gd-polynomial-minima.ipynb` | ⚠️ | ✅ `while` loop had no iteration cap; now bounded, with a grid-search cross-check on the minimum. |
| `linear-models/02-batch-gd-early-stopping-ridge.ipynb` | ❌ | ✅ `NameError` on `sel.patience`; was full-batch despite the "mini batch" title; L2 double-weighted. |
| `linear-models/03-polynomial-fourier-basis-regression.ipynb` | ❌ | ✅ Uninitialised counter; inverse normalisation computed `(max+min)+min`; design matrix rebuilt from data inside `predict`. |
| `linear-models/06-perceptron-iterative.ipynb` | ❌ | ✅ Swapped argument order; `derr` arity mismatch; aliasing; **both update branches identical**. |
| `linear-models/07-mlp-backprop-from-scratch.ipynb` | ❌ | ✅ Rewritten with explicit weight matrices. Gradients now match finite differences to 4e-10; 97.5% test accuracy. |
| `tree-models/01-decision-tree-information-gain.ipynb` | ❌ | ✅ `NameError` on `data_n`; `and col` made feature 0 unreachable; and the left/right children were **swapped relative to `predict`**. |
| `tree-models/02-decision-tree-visualisation.ipynb` | ❌ | ✅ `sklearn.externals.six` removed in 0.23; replaced with `io.StringIO`. |

---

## Working notebooks

### `01-linear-regression-gradient-descent.ipynb`

Derives linear regression by hand: normalise X and y to `[0, 1]`, write the
squared-error gradients with respect to each coefficient, average them over the
data, and iterate. Ends by inverting the normalisation to plot the fit on the
original scale.

The normalisation is not decoration. A single scalar learning rate applied to
poorly scaled features either crawls or diverges — the notebook's own comment
("you may need to slowly diminish that value") is recognising that.

Worth noting as good practice: it plots the weights *and* the error together,
so the convergence behaviour is visible rather than assumed.

### `05-perceptron-from-scratch.ipynb`

The perceptron update rule on 20,000 synthetic 21-dimensional binary samples,
tracking classification accuracy across epochs to watch it climb. The accuracy
trace is the point — it makes convergence observable.

### `04-momentum-gd-polynomial-minima.ipynb`

Gradient descent with momentum on a degree-8 polynomial, using an analytic
derivative computed from the coefficient vector. Walks the trajectory to the
minimum.

⚠️ The loop has no iteration cap. On an unlucky starting point it will not
terminate. The fix is a `max_iter` bound plus a break — the same correction
applied in the package.

---

## What was broken, and why it mattered

Kept as a record. Each of these was invisible without *running* the code, and
two of them — the disabled feature and the swapped children — produced no error
at all.

### `01-decision-tree-information-gain.ipynb` — feature 0 could never be used

```python
if ent[col][ind] < min_avg_ent and col:
```

`col` is the feature index. Column `0` is falsy in Python, so `and col` is
False whenever the tree considered the first feature — it could never be
selected. The tree was silently crippled and would have scored at chance on any
problem where column 0 mattered.

No error, no warning. Just a permanently worse model. `test_feature_zero_is_usable`
in the test suite exists solely to keep this from returning.

A second bug: `choose_best_feature` referenced `data_n` where it meant its own
`data` argument, so the function raised `NameError` on every call.

### `02-batch-gd-early-stopping-ridge.ipynb` — two problems

```python
patience_left = sel.patience + 1     # `sel` is never defined
```

The early-stopping loop cannot run. Separately, the notebook is titled "mini
batch GD" but computes a full-batch update — the batch size is never varied.
The title describes an algorithm the code does not implement.

The ridge/L1 blending term is also worth questioning:
`r*alpha*w + (alpha*(1-r)/2)*w` gives L2 twice the weight of L1 at equal `r`,
which is probably not what was intended.

### `03-polynomial-fourier-basis-regression.ipynb`

- `i += 1` inside `fit` with `i` never initialised → `NameError` on first pass.
- The inverse normalisation reads `X *= (max + min) + min`. The inverse of
  `(x - min) / (max - min)` is `(x * (max - min)) + min`. As written it does
  not invert anything.
- A deeper design problem: the design matrix is rebuilt from the *data* at
  prediction time. The basis is data-dependent, so the model is not actually a
  function of the weights alone, and `predict` on new points is unsound.

### `06-perceptron-iterative.ipynb`

- `y_pred(w, x)` is defined one way and called the other way round.
- `error(y, x, w, b)` takes four arguments; `derr(x)` calls it with three.
- `wt = w` then `wt[-1] = b` mutates `w` in place through the alias.
- Both branches of the update do the identical line:

  ```python
  if y_cap - y[i] == 1:
      w -= x @ learn_rate
  elif y_cap - y[i] == -1:
      w -= x @ learn_rate      # same line, different sign needed
  ```

  The sign is supposed to come from the branch. As written the rule cannot
  learn anything.

### `07-mlp-backprop-from-scratch.ipynb` — unfinished, and instructive

The most interesting failure here, because the *shape* of the problem is wrong
rather than a typo:

- `predict()` calls `argmax`, which is never imported or defined.
- `fit()` computes backpropagation and then returns nothing.
- The hidden-layer weight update
  `np.sum(y_hat.reshape(-1, 1) @ w.reshape(1, 1), axis=0)` produces a gradient
  with the wrong shape, and reverses the weight vector (`[::-1]`) before using
  it, so forward and backward disagree about layer order.
- The learning rate is not decoupled from loss scaling, so the effective step
  size drifts as training proceeds.

**The fix is structural, not a patch.** Store weights as explicit
`(n_in, n_out)` matrices per layer. Forward becomes `x @ W + b`; backward
becomes `W -= lr * (delta @ x.T)`. Neither the ordering bug nor the shape bug
can occur. Then verify with finite differences:

```python
# numerical gradient
eps = 1e-6
numeric = (loss(W + eps) - loss(W - eps)) / (2 * eps)
# assert np.allclose(numeric, analytic, atol=1e-6)
```

A gradient checker is the only way to be sure. The reasoning is worth reading;
the code is not correct.

### `02-decision-tree-visualisation.ipynb`

`from sklearn.externals.six import StringIO` — `sklearn.externals` was removed
in scikit-learn 0.23. Use `from io import StringIO`.

The `export_graphviz` approach itself is sound, and reaching into
`model.tree_.children_left` / `.feature` / `.threshold` is a good way to see how
a fitted tree is actually stored.

---

## What the package changed

| Original defect | Fix |
|---|---|
| `data_n` undefined | Argument used directly |
| `and col` disabled feature 0 | Guard removed; regression test added |
| In-place normalisation corrupted inputs | Non-mutating transforms; regression test added |
| `un_norm` arithmetic error | Exact inverse; round-trip test added |
| `sel.patience` undefined | Constructor parameter |
| `tol` documented but unused | Now gates early stopping |
| Mini-batch title, full-batch code | `batch_size` is a real parameter |
| Swapped argument order | Consistent signatures |
| Duplicate update branches | Single update driven by the error sign |
| Weight-shape confusion in backprop | Explicit `(n_in, n_out)` matrices |
| Unbounded `while` loops | Epoch budgets everywhere |

Run the tests with `pytest tests/ -v`. Each regression test is named after the
bug it prevents — `test_feature_zero_is_usable` exists for exactly one reason.
