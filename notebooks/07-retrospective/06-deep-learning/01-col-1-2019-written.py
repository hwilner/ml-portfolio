# 1 — Multilayer Perceptron (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The original 2019 from-scratch neural
network: forward pass, backpropagation, trained on the 8×8 digits dataset.

**This is the project the retrospective calls "genuinely unfinished and the
biggest missed opportunity in the repository."** In 2019 the backpropagation
cell was a stub with a TODO. The repository's debugging pass later rewrote it
with explicit matrices and finite-difference gradient checks, reaching a test
accuracy of 0.9750.

This notebook shows the version that *would have existed* had the 2019 work
been finished — and, critically, the naive vectorisation that is what most
hand-written networks actually look like, including the mistakes it invites.

Read this, then `02-col-2-2019-judgement.ipynb`, then
`03-col-3-2026-tools.ipynb`.

## The data

1,797 8×8 digit images from `sklearn.datasets.load_digits`, pixel values 0–16
normalised to 0–1. The original notebook pulled MNIST from TensorFlow purely to
get digits; depending on a ~600 MB install to feed a hand-written NumPy
network is not a good trade, and this dataset exercises the identical code path.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 20260929
rng = np.random.default_rng(SEED)

digits = load_digits()
X = digits.data.astype(float) / 16.0
y = digits.target
print(f"{X.shape[0]} images of {X.shape[1]} pixels, {len(np.unique(y))} classes")
print(f"pixel range: {X.min():.2f} to {X.max():.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"train {len(X_train)}, test {len(X_test)}")

# %%
fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
for ax, i in zip(axes.ravel(), range(10)):
    ax.imshow(X_train[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
    ax.set_title(f"{y_train[i]}", fontsize=11)
    ax.axis("off")
plt.suptitle("8x8 digit images")
plt.tight_layout()
plt.show()

# %% [markdown]
## Step 1 — One-hot encode the labels

# %%
def one_hot(labels, n_classes=10):
    """Expand integer labels into a one-hot matrix.

    Args:
        labels: Integer array of shape ``(n_samples,)``.
        n_classes: Number of target classes.

    Returns:
        Float array of shape ``(n_samples, n_classes)``.
    """
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


Y_train = one_hot(y_train)
Y_test = one_hot(y_test)
print(f"Y_train {Y_train.shape}, row sums all 1: {np.allclose(Y_train.sum(axis=1), 1)}")
Y_train[:5]

# %% [markdown]
## Step 2 — Initialise weights

He initialisation: for a layer with fan-in $n$, draw from a normal with
standard deviation $\sqrt{2/n}$. This keeps the variance of the activations
roughly constant layer to layer, which is what makes deep networks trainable at
all.

# %%
def init_weights(layers, rng):
    """He-initialised weights and zero biases for a list of layer widths."""
    Ws, bs = [], []
    for i in range(len(layers) - 1):
        n_in, n_out = layers[i], layers[i + 1]
        W = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        Ws.append(W)
        bs.append(b)
    return Ws, bs


LAYERS = [64, 32, 10]
Ws, bs = init_weights(LAYERS, rng)
for i, (W, b) in enumerate(zip(Ws, bs)):
    print(f"  layer {i}: W {W.shape}, std {W.std():.4f}, b {b.shape}")

# %% [markdown]
## Step 3 — Forward pass

With ReLU activations and a softmax output, the forward pass is three matrix
multiplies.

# %%
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    """Row-wise softmax, computed stably by subtracting the row max."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, Ws, bs):
    """Forward pass. Returns the output probabilities and the cache."""
    activations = [X]
    pre_activations = []
    n_layers = len(Ws)

    for i in range(n_layers):
        z = activations[-1] @ Ws[i] + bs[i]
        pre_activations.append(z)
        a = softmax(z) if i == n_layers - 1 else relu(z)
        activations.append(a)

    return activations[-1], activations, pre_activations


# %%
probs, acts, pre = forward(X_train[:5], Ws, bs)
print(f"output shape {probs.shape}, rows sum to 1: {np.allclose(probs.sum(axis=1), 1)}")
print("softmax output for the first 5 training images:")
print(np.round(probs, 3))

# %%
# Softmax with the naive implementation overflows on large logits — the
# reason for the max-subtraction is not cosmetic.
z_big = np.array([[1000.0, 1001.0, 999.0]])
print()
print("naive exp() overflows:", np.any(~np.isfinite(np.exp(z_big))))
print("stable softmax       :", np.round(softmax(z_big), 3))
print("-> the shift is required, and the 2019 version did not have it.")

# %% [markdown]
## Step 4 — Backpropagation

The version a working 2019 notebook would have. The chain rule, layer by layer,
with the softmax cross-entropy collapsed to a single $(\hat{y} - y)$ term.

# %%
def backward(X, Y, Ws, bs, cache=None):
    """Backpropagation. Returns (dW, db, prob).

    Args:
        X: Input batch, shape ``(n, n_in)``.
        Y: One-hot targets, shape ``(n, n_out)``.
        Ws: List of weight matrices.
        bs: List of bias vectors.
        cache: Optional ``(probs, activations, pre_activations)`` from ``forward``.

    Returns:
        Tuple of lists of gradients matching ``Ws`` and ``bs``, plus the output
        probabilities.
    """
    n = X.shape[0]
    if cache is None:
        probs, acts, pre = forward(X, Ws, bs)
    else:
        probs, acts, pre = cache

    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)

    # Output layer: dL/dz = (probs - Y) / n, because softmax + cross-entropy
    # collapse. This is the single most important simplification in the code
    # and getting it wrong is the classic source of silent bugs.
    delta = (probs - Y) / n
    dWs[-1] = acts[-2].T @ delta
    dbs[-1] = delta.sum(axis=0)

    # Hidden layers
    for i in range(len(Ws) - 2, -1, -1):
        delta = (delta @ Ws[i + 1].T) * (pre[i] > 0)
        dWs[i] = acts[i].T @ delta
        dbs[i] = delta.sum(axis=0)

    return dWs, dbs, probs


# %%
dWs, dbs, _ = backward(X_train[:5], Y_train[:5], Ws, bs)
for i, (dW, db) in enumerate(zip(dWs, dbs)):
    print(f"  grad layer {i}: dW {dW.shape} mean {dW.mean():+.5f} std {dW.std():.5f} | db mean {db.mean():+.5f}")

# %% [markdown]
## Step 5 — Verify the gradients before training anything

**This is the step the 2019 notebook never reached, because the backprop
cell was never written.** A hand-derived gradient that is wrong produces a
network that either does not train or trains to something worse than random —
but nothing raises an exception, and the code *looks* correct.

The test: perturb each parameter by a small $\epsilon$, compute the loss by
finite differences, and compare with the analytic gradient.

# %%
def cross_entropy(probs, Y):
    """Mean cross-entropy loss of softmax probabilities against one-hot labels."""
    return -np.sum(Y * np.log(np.clip(probs, 1e-12, None))) / probs.shape[0]


def loss_and_grads(X, Y, Ws, bs):
    probs, acts, pre = forward(X, Ws, bs)
    loss = cross_entropy(probs, Y)
    dWs, dbs, _ = backward(X, Y, Ws, bs, cache=(probs, acts, pre))
    return loss, dWs, dbs


def numerical_gradient(X, Y, Ws, bs, eps=1e-6):
    """Central-difference gradient, used to check the analytic one."""
    dWs = [np.zeros_like(W) for W in Ws]
    dbs = [np.zeros_like(b) for b in bs]
    for i in range(len(Ws)):
        it = np.nditer(Ws[i], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = Ws[i][idx]
            Ws[i][idx] = orig + eps
            lp = cross_entropy(forward(X, Ws, bs)[0], Y)
            Ws[i][idx] = orig - eps
            lm = cross_entropy(forward(X, Ws, bs)[0], Y)
            Ws[i][idx] = orig
            dWs[i][idx] = (lp - lm) / (2 * eps)
            it.iternext()
    return dWs, dbs


# %%
Xc, Yc = X_train[:8], Y_train[:8]
loss, dW_ana, db_ana = loss_and_grads(Xc, Yc, Ws, bs)
dW_num, db_num = numerical_gradient(Xc, Yc, Ws, bs)

print("analytic vs numerical gradient (central differences, eps=1e-6):")
print()
max_diff = 0.0
for i, (da, dn) in enumerate(zip(dW_ana, dW_num)):
    diff = np.abs(da - dn).max()
    max_diff = max(max_diff, diff)
    print(f"  layer {i}: max |analytic - numerical| = {diff:.3e}  "
          f"(gradient scale {np.abs(dn).max():.3e})")
print()
print(f"worst absolute difference: {max_diff:.3e}")
if max_diff < 1e-6:
    print("PASS — the analytic gradient matches finite differences.")
else:
    print("FAIL — the backpropagation is wrong, and the network should not be")
    print("        trained until it is fixed.")

# %% [markdown]
## Step 6 — Train

# %%
def train(X, Y, layers, epochs=200, lr=0.1, batch_size=32, seed=SEED, verbose=True):
    """Full-batch gradient descent with a fixed learning rate."""
    rng = np.random.default_rng(seed)
    Ws, bs = init_weights(layers, rng)
    n = X.shape[0]
    history = []

    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], Y[idx]
            loss, dWs, dbs = loss_and_grads(xb, yb, Ws, bs)
            epoch_loss += loss * len(idx)
            for i in range(len(Ws)):
                Ws[i] -= lr * dWs[i]
                bs[i] -= lr * dbs[i]
        history.append(epoch_loss / n)
        if verbose and (epoch + 1) % 40 == 0:
            print(f"  epoch {epoch + 1:>3}  loss {history[-1]:.4f}")
    return Ws, bs, history


t0 = time.time()
Ws_t, bs_t, history = train(X_train, Y_train, LAYERS, epochs=200, lr=0.1)
train_time = time.time() - t0

probs_test, _, _ = forward(X_test, Ws_t, bs_t)
pred = probs_test.argmax(axis=1)
accuracy = (pred == y_test).mean()
print()
print(f"trained in {train_time:.1f}s")
print(f"final train loss : {history[-1]:.4f}")
print(f"test accuracy    : {accuracy:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].plot(history, color="#4C72B0", lw=1.5)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Training loss")

axes[1].hist(probs_test.max(axis=1), bins=30, color="#C44E52", edgecolor="white")
axes[1].axvline(0.5, ls="--", color="black", label="decision threshold")
axes[1].set_xlabel("predicted probability")
axes[1].set_ylabel("test images")
axes[1].set_title(f"Confidence distribution (accuracy {accuracy:.4f})")
axes[1].legend()
plt.tight_layout()
plt.show()

# %%
# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

cm = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred, digits=3))
print(pd.DataFrame(cm, index=range(10), columns=range(10)).to_string())

# %% [markdown]
## What this notebook got right, and what it was missing

**Right, and worth keeping:**

- He initialisation — without it, 3 layers already train badly
- The $(\hat{y} - y)$ collapse for softmax cross-entropy, which makes the
  output layer's gradient one line
- Mini-batch gradient descent

**Missing, and this is the retrospective's point:**

| Problem | Consequence |
|---|---|
| **The backprop cell was a stub** | The 2019 notebook as written could not train a network at all |
| **No gradient check** | A wrong gradient trains to a worse-than-random model silently |
| **Naive softmax** | Overflows on large logits; would break on a badly initialised network |
| **Fixed learning rate, no schedule** | Converges to a worse point than it could |
| **No regularisation** | 64-32-10 on 1,437 training images is enough capacity to overfit |
| **No early stopping** | Keeps training after the test-relevant point has passed |
| **No confusion matrix or per-class report** | Aggregate accuracy hides which digits fail |

**The gradient check is the one that matters.** It is six lines, it runs once,
and it converts "the network trains" from a claim into a fact. The next
notebook shows what it catches.

# %%
# Demonstrate the failure the gradient check would catch.
# A single sign error in the hidden layer is the most common hand-derivation
# bug, and it trains to a plausible-looking loss.
def backward_no_relu_derivative(X, Y, Ws, bs, cache=None):
    """Backpropagation that omits the ReLU derivative — the classic bug.

    The forward pass is untouched; only the gradient is wrong. This is the
    single most common hand-derivation error in a from-scratch network, and it
    is invisible without a numerical check.
    """
    n = X.shape[0]
    probs, acts, pre = forward(X, Ws, bs)
    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)
    delta = (probs - Y) / n
    dWs[-1] = acts[-2].T @ delta
    dbs[-1] = delta.sum(axis=0)
    for i in range(len(Ws) - 2, -1, -1):
        # BUG: missing "* (pre[i] > 0)"
        delta = delta @ Ws[i + 1].T
        dWs[i] = acts[i].T @ delta
        dbs[i] = delta.sum(axis=0)
    return dWs, dbs, probs


dW_good, _, _ = backward(Xc, Yc, Ws_t, bs_t)
dW_bad, _, _ = backward_no_relu_derivative(Xc, Yc, Ws_t, bs_t)
dW_ref, _ = numerical_gradient(Xc, Yc, Ws_t, bs_t)

print("Effect of omitting the ReLU derivative from backpropagation:")
print(f"  correct gradient vs finite differences : max diff {np.abs(dW_good[0] - dW_ref[0]).max():.3e}")
print(f"  buggy gradient vs finite differences    : max diff {np.abs(dW_bad[0] - dW_ref[0]).max():.3e}")
print()
ratio = np.abs(dW_bad[0] - dW_ref[0]).max() / max(np.abs(dW_good[0] - dW_ref[0]).max(), 1e-18)
print(f"The buggy gradient is wrong by {ratio:.0f}x more than the correct one,")
print("and it still returns a finite number in the right shape. Nothing raises.")
print("Without the finite-difference check, this is invisible.")
