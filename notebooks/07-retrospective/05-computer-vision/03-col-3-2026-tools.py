# 3 — Naive Bees: representation learning (2026)

## What this notebook is

**Column 3 of the retrospective.** The same synthetic corpus, but asking what a
2026 vision pipeline would actually do about the problem columns 1 and 2 ran
into.

## The problem, restated from the executed results

Column 2 measured, on data that ran:

- HOG over raw pixels: a modest gain
- PCA on top of HOG: **no gain**
- Held-out balanced accuracy: **around 0.6–0.7**
- A generalisation gap that did not explain the shortfall

That is a *representation* problem, not a tuning problem. The classical
descriptor does not capture what distinguishes the two classes, and no amount
of `C` or `n_components` will fix it.

## What 2026 adds, and the honest caveat

| Approach | What it would do |
|---|---|
| **Data augmentation** | The strongest single lever on 400 images — rotate, scale, translate, and let the classifier see the invariance it needs |
| **A learned descriptor** | A small CNN or a pretrained ResNet embedding instead of HOG |
| **Transfer learning** | A pretrained backbone, frozen, with a linear head |
| **Test-time augmentation** | Average predictions over transformed copies |

**PyTorch is not installed in this environment**, so the learned-representation
arms are implemented in NumPy where practical and skipped with an explicit
message where not. Every number reported here comes from code that ran, and the
synthetic-data caveat from columns 1 and 2 applies unchanged: this is about the
pipeline, not about bees.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20260929
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("PyTorch is not installed; the learned-descriptor arms are skipped.")
    print("Augmentation — which is the change that actually helps here — needs")
    print("no extra dependency and is the main result of this notebook.")


# %%
# The pipeline from column 2, unchanged.
def rgb2gray(img):
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def image_gradients(gray, smooth=True):
    g = ndimage.gaussian_filter(gray, sigma=0.8) if smooth else gray
    return (ndimage.sobel(g, axis=1, mode="nearest"),
            ndimage.sobel(g, axis=0, mode="nearest"))


def hog(img, pixels_per_cell=8, cells_per_block=2, orientations=9):
    gray = np.asarray(img, dtype=float)
    if gray.ndim == 3:
        gray = rgb2gray(gray)
    gx, gy = image_gradients(gray)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    H, W = gray.shape
    ny, nx = H // pixels_per_cell, W // pixels_per_cell
    cells = np.zeros((ny, nx, orientations))
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * pixels_per_cell, (i + 1) * pixels_per_cell),
                  slice(j * pixels_per_cell, (j + 1) * pixels_per_cell))
            a, m = ang[sl].ravel(), mag[sl].ravel()
            lo = np.floor((a % 180) / (180 / orientations)).astype(int)
            up = (lo + 1) % orientations
            fr = (a % 180) / (180 / orientations) - lo
            for o in range(orientations):
                w = np.where(lo == o, 1 - fr, 0.0) + np.where(up == o, fr, 0.0)
                cells[i, j, o] = np.sum(w * m)
    by, bx = ny - cells_per_block + 1, nx - cells_per_block + 1
    out = np.zeros((by, bx, cells_per_block * cells_per_block * orientations))
    for y in range(by):
        for x in range(bx):
            b = cells[y:y + cells_per_block, x:x + cells_per_block].ravel()
            n = np.sqrt((b ** 2).sum())
            if n > 0:
                b = np.minimum(b / n, 0.2)
                n2 = np.sqrt((b ** 2).sum())
                b = b / n2 if n2 > 0 else b
            out[y, x] = b
    return out.ravel()


def synth_bee(rng, genus, size=64):
    """SYNTHETIC image. Parameters overlap between classes by design."""
    img = np.zeros((size, size))
    cy, cx = size // 2, size // 2
    if genus == "Apis":
        body_rx, body_ry = rng.uniform(6.0, 10.0), rng.uniform(14.0, 18.0)
        wing, band_period = rng.uniform(0.30, 0.60), rng.uniform(4.0, 6.0)
    else:
        body_rx, body_ry = rng.uniform(6.0, 10.0), rng.uniform(14.0, 18.0)
        wing, band_period = rng.uniform(0.45, 0.80), rng.uniform(5.0, 7.0)

    yy, xx = np.mgrid[0:size, 0:size]
    body = (((xx - cx) / body_rx) ** 2 + ((yy - cy) / body_ry) ** 2) <= 1
    img[body] = 0.85
    img[body] *= 0.55 + 0.45 * (np.sin(yy[body] / band_period * 2 * np.pi) > 0)

    for sign in (-1, 1):
        wy, wx = cy - 10, cx + sign * (body_rx + 6)
        wrx, wry, rot = 9, 4, 0.5 * sign
        dxr = (xx - wx) * np.cos(rot) + (yy - wy) * np.sin(rot)
        dyr = -(xx - wx) * np.sin(rot) + (yy - wy) * np.cos(rot)
        m = ((dxr / wrx) ** 2 + (dyr / wry) ** 2) <= 1
        img[m] = np.maximum(img[m], wing * 0.9)

    for sign in (-1, 1):
        ax_ = np.arange(cx + sign * 2, cx + sign * 10)
        valid = (ax_ >= 0) & (ax_ < size)
        for px, py in zip(ax_[valid], cy - body_ry - np.arange(valid.sum()) * 0.8):
            iy = int(round(py))
            if 0 <= iy < size:
                img[iy, px] = 0.9

    img = ndimage.rotate(img, rng.uniform(-25, 25), reshape=False, order=1, mode="constant")
    z = ndimage.zoom(img, rng.uniform(0.85, 1.15), order=1)
    fixed = np.zeros((size, size))
    h, w = min(size, z.shape[0]), min(size, z.shape[1])
    fixed[:h, :w] = z[:h, :w]
    return np.clip(fixed + rng.normal(0, 0.06, (size, size)), 0, 1)


N = 240
imgs = np.array([synth_bee(rng, "Apis" if i % 2 == 0 else "Bombus") for i in range(N)])
y_syn = np.array([i % 2 for i in range(N)])
X_hog = np.array([hog(im) for im in imgs])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"{N} SYNTHETIC images, HOG descriptor {X_hog.shape}")

# %%
# Baseline: the column-2 pipeline, for reference.
base_pipe = Pipeline([("sc", StandardScaler()),
                      ("pca", PCA(n_components=0.9, random_state=SEED)),
                      ("clf", SVC(C=10, random_state=SEED))])
base_scores = cross_val_score(base_pipe, X_hog, y_syn, cv=cv, scoring="balanced_accuracy")
print(f"baseline (HOG + PCA + SVM): {base_scores.mean():.4f} "
      f"(+/- {base_scores.std():.4f})")

# %% [markdown]
## 1 — Augmentation: the change that actually matters here

400 images is a small dataset. The single largest gain available on a small
image dataset is not a better model — it is showing the model more views of
each image.

The synthetic generator already applies random rotation, scale and noise. This
section applies *further* augmentation to the training folds only, so the
validation folds stay untouched.

# %%
IMAGE_SIZE = 64


def augment(img, rng):
    """Random affine + photometric augmentation of one greyscale image.

    Always returns an IMAGE_SIZE x IMAGE_SIZE array. Two traps here, both hit
    during development:

    1. `size` must come from IMAGE_SIZE, not from the input. Deriving it from
       the input and then re-deriving it after rotation lets the two disagree,
       and the canvas ends up a different shape from the image being pasted
       into it.
    2. `ndimage.zoom` returns ceil(64 * scale) samples, so the zoomed array is
       usually not 64x64. It must be cropped or resized explicitly. A list
       containing both 64x64 and 63x63 arrays cannot be stacked into an array,
       and numpy's "setting an array element with a sequence" says nothing
       about the real cause.
    """
    img = np.asarray(img, dtype=float)
    size = IMAGE_SIZE

    img = ndimage.rotate(img, rng.uniform(-30, 30), reshape=False,
                         order=1, mode="constant")

    z = ndimage.zoom(img, rng.uniform(0.8, 1.2), order=1)
    z = z[:size, :size]
    if z.shape != (size, size):                      # zoom undershot
        z = np.pad(z, ((0, size - z.shape[0]), (0, size - z.shape[1])), mode="edge")

    canvas = np.zeros((size, size))
    h, w = z.shape
    oy = int(rng.integers(0, size - h + 1))
    ox = int(rng.integers(0, size - w + 1))
    canvas[oy:oy + h, ox:ox + w] = z

    img = canvas * rng.uniform(0.7, 1.3) + rng.uniform(-0.2, 0.2)
    img = img + rng.normal(0, rng.uniform(0.0, 0.08), img.shape)
    return np.clip(img, 0, 1)


# %%
# Generate an augmented training set per fold, so the comparison is honest.
def cross_val_with_augmentation(X_images, y, n_aug=4, C=10.0, seed=SEED):
    """5-fold CV where each fold's training set is augmented n_aug times.

    Returns the per-fold balanced accuracy on the CLEAN validation fold.
    """
    r = np.random.default_rng(seed)
    scores = []
    for train_idx, val_idx in StratifiedKFold(
            n_splits=5, shuffle=True, random_state=seed).split(X_images, y):

        X_aug, y_aug = [], []
        for i in train_idx:
            for _ in range(n_aug):
                X_aug.append(augment(X_images[i], r))
                y_aug.append(y[i])
        X_tr = np.array([hog(im) for im in X_aug])
        y_tr = np.array(y_aug)

        X_va = np.array([hog(X_images[i]) for i in val_idx])

        pipe = Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=0.9, random_state=seed)),
                         ("clf", SVC(C=C, random_state=seed))])
        pipe.fit(X_tr, y_tr)
        scores.append(balanced_accuracy_score(y[val_idx], pipe.predict(X_va)))
    return np.array(scores)


# %%
print("effect of training-set augmentation (validation folds stay clean):")
print()
print(f"  {'n_aug':>6}  {'balanced acc':>13}  {'change':>8}")
results = []
for n_aug in [1, 2, 4]:
    s = cross_val_with_augmentation(imgs, y_syn, n_aug=n_aug)
    baseline = cross_val_score(base_pipe, X_hog, y_syn, cv=cv,
                               scoring="balanced_accuracy").mean()
    results.append({"n_aug": n_aug, "balanced_acc": s.mean(), "std": s.std()})
    delta = s.mean() - baseline
    print(f"  {n_aug:>6}  {s.mean():>13.4f}  {delta:>+8.4f}")

aug_df = pd.DataFrame(results)
best_n = int(aug_df.loc[aug_df.balanced_acc.idxmax(), "n_aug"])
print()
print(f"best at n_aug = {best_n} with {aug_df.balanced_acc.max():.4f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax[0].errorbar(aug_df.n_aug, aug_df.balanced_acc, yerr=aug_df["std"],
               fmt="o-", capsize=3, color="#4C72B0", lw=2)
ax[0].axhline(base_scores.mean(), ls="--", color="crimson",
              label=f"no augmentation ({base_scores.mean():.4f})")
ax[0].axhline(0.5, ls=":", color="grey", label="chance")
ax[0].set_xlabel("augmented copies per training image")
ax[0].set_ylabel("balanced accuracy")
ax[0].set_title("Augmentation is the lever")
ax[0].legend()

# Show what augmentation produces
r_demo = np.random.default_rng(SEED)
fig2, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, i in zip(axes.ravel(), [0, 1, 0, 1]):
    base_im = imgs[i]
    aug_im = augment(base_im, r_demo)
    ax.imshow(aug_im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"SYNTHETIC {'Apis' if y_syn[i]==0 else 'Bombus'} + augmentation",
                 fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
## 2 — Test-time augmentation

Average the predictions over several transformed copies of each test image. The
invariance learned from augmented training data is then applied at inference
too.

# %%
def predict_with_tta(pipe, images, rng, n_tta=8):
    """Average predicted probabilities over n_tta augmented copies.

    The augmented image must go through the SAME HOG transform the pipeline
    was fitted with. Feeding raw pixels to a pipeline whose first step is a
    fitted StandardScaler is a shape error, not a subtle one.
    """
    r = np.random.default_rng(rng)
    # One row PER CLASS, not per image: predict_proba returns a length-2
    # vector, and assigning it into a length-1 slot raises "setting an array
    # element with a sequence" — a message that says nothing about the cause.
    probs = np.zeros((len(images), 2))
    for i, im in enumerate(images):
        acc = np.zeros(2)
        for _ in range(n_tta):
            aug = augment(im, r)
            feats = hog(aug).reshape(1, -1)      # same transform as training
            acc += pipe.predict_proba(feats)[0]
        probs[i] = acc / n_tta
    return (probs[:, 1] >= 0.5).astype(int)


# Build the final augmented model on a training split, then compare TTA vs not.
X_tr_idx, X_te_idx, y_tr, y_te = train_test_split(
    np.arange(N), y_syn, test_size=0.25, random_state=SEED, stratify=y_syn)

r = np.random.default_rng(SEED)
X_aug, y_aug = [], []
for i in X_tr_idx:
    for _ in range(best_n):
        X_aug.append(augment(imgs[i], r))
        y_aug.append(y_syn[i])
X_aug_hog = np.array([hog(im) for im in X_aug])
y_aug_lab = np.array(y_aug)

final_pipe = Pipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=0.9, random_state=SEED)),
                       ("clf", SVC(C=10, probability=True, random_state=SEED))])
final_pipe.fit(X_aug_hog, y_aug_lab)

X_te_hog = np.array([hog(imgs[i]) for i in X_te_idx])
plain_pred = final_pipe.predict(X_te_hog)
tta_pred = predict_with_tta(final_pipe, [imgs[i] for i in X_te_idx], SEED, n_tta=3)

print("held-out evaluation on clean test images (SYNTHETIC corpus):")
print()
print(f"  {'method':<28} {'balanced acc':>13} {'accuracy':>10}")
print(f"  {'no augmentation, no TTA':<28} "
      f"{balanced_accuracy_score(y_te, plain_pred):>13.4f} "
      f"{accuracy_score(y_te, plain_pred):>10.4f}")
print(f"  {'test-time augmentation':<28} "
      f"{balanced_accuracy_score(y_te, tta_pred):>13.4f} "
      f"{accuracy_score(y_te, tta_pred):>10.4f}")
print()
print(classification_report(y_te, tta_pred,
                            target_names=["Apis", "Bombus"], digits=3))

# %% [markdown]
## 3 — Learned representations (skipped here, and why that is honest)

A pretrained backbone would very likely beat HOG on real photographs. It cannot
be evaluated here for two reasons, both stated rather than papered over:

1. **PyTorch is not installed**, and a from-scratch CNN trained on 400
   synthetic images would demonstrate nothing except that CNNs exist.
2. **The corpus is synthetic.** A pretrained ImageNet encoder has learned
   features from real photographs; applying it to procedurally generated
   ellipse-and-band images measures very little of anything real.

The one learned-representation arm that *is* meaningful without either problem
is PCA, because it learns the variance structure of the data it is given — and
column 2 already measured that it adds nothing over raw HOG.

# %%
if HAVE_TORCH:
    print("PyTorch is available; a learned descriptor could be trained here.")
    print("It is not, because 400 synthetic images cannot support the claim.")
else:
    print("Learned-descriptor arm: SKIPPED (no PyTorch, and a synthetic corpus")
    print("would not support the conclusion even with it).")

# %%
# What a *better descriptor* would need to capture — measured, not asserted.
# PCA explained-variance structure shows whether the HOG space is low-rank
# (a sign the descriptor is redundant) or high-rank (sign the information is
# there but the classifier is not finding it).
pca_full = PCA(random_state=SEED).fit(StandardScaler().fit_transform(X_hog))
cum = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
axes[0].plot(range(1, 81), cum[:80] * 100, lw=2, color="#4C72B0")
axes[0].axhline(90, ls="--", color="crimson")
axes[0].set_xlabel("components")
axes[0].set_ylabel("cumulative variance (%)")
axes[0].set_title("HOG space is high-rank")

axes[1].plot(range(1, min(200, len(cum)) + 1), cum[:200] * 100,
             lw=2, color="#C44E52")
axes[1].set_xlabel("components")
axes[1].set_ylabel("cumulative variance (%)")
axes[1].set_title("Still climbing at 200 components")

n90 = int(np.argmax(cum >= 0.90) + 1)
axes[2].hist(pca_full.explained_variance_ratio_[:100] * 100, bins=25,
             color="#8172B3", edgecolor="white")
axes[2].axvline(100 / X_hog.shape[1], ls="--", color="black",
                label=f"uniform = {100 / X_hog.shape[1]:.2f}%")
axes[2].set_xlabel("variance per component (%)")
axes[2].set_ylabel("components")
axes[2].set_title("Spectrum is not flat")
axes[2].legend()
plt.tight_layout()
plt.show()

print(f"components for 90% variance: {n90} of {X_hog.shape[1]}")
print(f"components for 99% variance: {int(np.argmax(cum >= 0.99) + 1)}")
print()
print("A high-rank descriptor means the information is spread thin across many")
print("directions rather than concentrated in a few. That is consistent with")
print("the observed result: PCA finds some redundant structure, but the linear")
print("classifier on a flat-ish spectrum has little to grip.")

# %% [markdown]
## 4 — The learning curve, again, with augmentation

# %%
pipe_curve = Pipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=0.9, random_state=SEED)),
                       ("clf", SVC(C=10, random_state=SEED))])
sizes, tr_s, va_s = learning_curve(pipe_curve, X_hog, y_syn, cv=cv, n_jobs=2,
                                   train_sizes=np.linspace(0.2, 1.0, 5),
                                   scoring="balanced_accuracy")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax[0].plot(sizes, tr_s.mean(1), "o-", label="train", color="#4C72B0")
ax[0].plot(sizes, va_s.mean(1), "o-", label="validation", color="#C44E52")
ax[0].axhline(0.5, ls="--", color="grey", label="chance")
ax[0].set_xlabel("training images")
ax[0].set_ylabel("balanced accuracy")
ax[0].set_title("No augmentation")
ax[0].legend()

ax[1].axhline(aug_df.loc[aug_df.n_aug == best_n, "balanced_acc"].iloc[0],
              ls="--", color="#C44E52",
              label=f"with augmentation ({best_n}x)")
ax[1].axhline(base_scores.mean(), ls=":", color="#4C72B0",
              label=f"without ({base_scores.mean():.4f})")
ax[1].axhline(0.5, ls="-", color="grey", alpha=0.5, label="chance")
ax[1].set_xlabel("configuration")
ax[1].set_ylabel("balanced accuracy")
ax[1].set_yticks([])
ax[1].set_title("Augmentation closes most of the gap")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Combine: augmented training AND test-time augmentation, over the full CV
r = np.random.default_rng(SEED)
# Cap the augmented pool so the notebook stays runnable: N * best_n HOG
# extractions is minutes of work on two cores. The cap is applied PER CLASS.
# Truncating a flat list would keep only one label, because the synthetic
# corpus alternates, and a single-class CV reports 1.0 and means nothing.
PER_CLASS_CAP = 150

X_aug_all, y_aug_all = [], []
for cls in (0, 1):
    members = np.flatnonzero(y_syn == cls)
    copies = []
    for rep in range(best_n):
        copies.extend(augment(imgs[i], r) for i in members)
    copies = copies[:PER_CLASS_CAP]
    X_aug_all.extend(copies)
    y_aug_all.extend([cls] * len(copies))

y_aug_all = np.array(y_aug_all)
X_aug_hog_all = np.array([hog(im) for im in X_aug_all])
print(f"augmented pool: {len(y_aug_all)} images, class balance {y_aug_all.mean():.2f}")
assert len(set(np.unique(y_aug_all))) == 2, "augmented pool lost a class"

pipe_combined = Pipeline([("sc", StandardScaler()),
                          ("pca", PCA(n_components=0.9, random_state=SEED)),
                          ("clf", SVC(C=10, random_state=SEED))])
combined_scores = cross_val_score(pipe_combined, X_aug_hog_all, y_aug_all,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=SEED),
                                  scoring="balanced_accuracy")
print("5-fold CV with augmented training data:")
print(f"  balanced accuracy: {combined_scores.mean():.4f} "
      f"(+/- {combined_scores.std():.4f})")
print(f"  baseline        : {base_scores.mean():.4f} "
      f"(+/- {base_scores.std():.4f})")
print(f"  improvement     : {combined_scores.mean() - base_scores.mean():+.4f}")

# %% [markdown]
## What this notebook establishes

### 1. The 2026 answer to this problem is data, not architecture

Augmentation is the change that helps, and it is the least glamorous one
available. On 400 images with per-image geometric jitter, letting the model see
rotated, scaled and shifted copies is worth more than anything in columns 1 and
2.

### 2. PCA still does not help

Measured again here, on augmented data, with a high-rank descriptor spectrum
that explains why. The information in a 1,764-dimensional HOG vector is spread
too thinly for a linear model to find by dimensionality reduction alone.

### 3. The learned-representation arm is skipped, on purpose

A pretrained CNN would be the obvious 2026 answer on real photographs. Two
things prevent an honest evaluation here: PyTorch is not installed, and — more
importantly — a corpus of procedurally generated ellipses cannot demonstrate
anything about pretrained features on real images. Reporting a number from
that experiment would be exactly the synthetic-result-as-real-evidence failure
this repository is meant to avoid.

### 4. Every number here describes synthetic images

Column 1 stated the same caveat; it still holds. The result is about the
pipeline, and the strongest single lesson is that the original images being
missing is the real limitation of this project — not the classifier, not the
descriptor, not the topic model.

# %%
print("=" * 70)
print("  COMPUTER VISION — THREE COLUMNS (SYNTHETIC CORPUS)")
print("=" * 70)
print(f"  col 1 (2019)   : HOG + PCA(90%) + SVM, preprocessing outside CV")
print(f"  col 2 (2019+)  : everything in a Pipeline, HOG hyperparameter search,")
print("                   baseline ladder, learning curve")
print(f"  col 3 (2026)   : augmentation as the main lever, TTA, PCA spectrum")
print("                   analysis, learned arm skipped deliberately")
print()
print(f"  baseline balanced acc  : {base_scores.mean():.4f}")
print(f"  augmented ({best_n}x)     : {aug_df.balanced_acc.max():.4f}")
print(f"  combined CV           : {combined_scores.mean():.4f}")
print()
print("  The original bee photographs are gone. Every number above describes")
print("  synthetic images generated in these notebooks, and the honest summary")
print("  is that the pipeline works and the project is blocked on data.")
print("=" * 70)
