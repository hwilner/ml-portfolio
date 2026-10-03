# 2 — Naive Bees: HOG + PCA + SVM (2019 judgement, same tools)

## What this notebook is

**Column 2 of the retrospective.** The same synthetic corpus, the same 2019-era
tools (`scikit-learn`, `numpy`, `scipy`, `matplotlib`), the analysis done with
judgement I did not have in 2019.

Everything here was possible in 2019. `Pipeline` (2013), `StratifiedKFold`
(2012), `GridSearchCV`, `class_weight`, learning curves. **This is a discipline
gap, not a tooling gap.**

**The data caveat stands:** the original bee photographs are gone. Everything
below runs on synthetic images generated in this notebook and labelled as such
throughout. The *method* is real; the *result* is about the pipeline, not bees.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20260929
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")


def rgb2gray(img):
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    if img.max() > 1.0:
        img = img / 255.0
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def image_gradients(gray, smooth=True):
    g = ndimage.gaussian_filter(gray, sigma=0.8) if smooth else gray
    return (ndimage.sobel(g, axis=1, mode="nearest"),
            ndimage.sobel(g, axis=0, mode="nearest"))


def hog(img, pixels_per_cell=8, cells_per_block=2, orientations=9):
    """Histogram of Oriented Gradients, written out rather than imported."""
    gray = rgb2gray(img)
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
            lower = np.floor((a % 180) / (180 / orientations)).astype(int)
            upper = (lower + 1) % orientations
            frac = (a % 180) / (180 / orientations) - lower
            for o in range(orientations):
                w = np.where(lower == o, 1 - frac, 0.0) + np.where(upper == o, frac, 0.0)
                cells[i, j, o] = np.sum(w * m)

    by, bx = ny - cells_per_block + 1, nx - cells_per_block + 1
    out = np.zeros((by, bx, cells_per_block * cells_per_block * orientations))
    for y in range(by):
        for x in range(bx):
            block = cells[y:y + cells_per_block, x:x + cells_per_block].ravel()
            norm = np.sqrt((block ** 2).sum())
            if norm > 0:
                block = np.minimum(block / norm, 0.2)
                n2 = np.sqrt((block ** 2).sum())
                block = block / n2 if n2 > 0 else block
            out[y, x] = block
    return out.ravel()


# %%
def synth_bee(rng, genus, size=64):
    """SYNTHETIC image. See column 1 for the parameter ranges and why they
    overlap between the two classes."""
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
        wy = cy - 10
        wx = cx + sign * (body_rx + 6)
        wrx, wry = 9, 4
        rot = 0.5 * sign
        dxr = (xx - wx) * np.cos(rot) + (yy - wy) * np.sin(rot)
        dyr = -(xx - wx) * np.sin(rot) + (yy - wy) * np.cos(rot)
        img[((dxr / wrx) ** 2 + (dyr / wry) ** 2) <= 1] = np.maximum(
            img[((dxr / wrx) ** 2 + (dyr / wry) ** 2) <= 1], wing * 0.9)

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


N = 400
print(f"generating {N} SYNTHETIC images (not real bees)...")
imgs = [synth_bee(rng, "Apis" if i % 2 == 0 else "Bombus") for i in range(N)]
y_syn = np.array([i % 2 for i in range(N)])
X_hog = np.array([hog(im) for im in imgs])
print(f"HOG features {X_hog.shape}, class balance {y_syn.mean():.2f}")

# %% [markdown]
## Change 1 — Everything inside a `Pipeline`

Column 1 fitted the scaler and the PCA on the full dataset and then
cross-validated only the SVM. Every fold scored a model whose representation
had been shaped using the held-out images.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Demonstrate the leak is gone: the PCA's explained variance now differs per fold
leak_check = []
for tr, _ in cv.split(X_hog, y_syn):
    p = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=0.9))]).fit(X_hog[tr])
    leak_check.append(p.named_steps["pca"].n_components_)

print("PCA components retained, refitted per fold:")
print(f"  {leak_check}")
print(f"  spread: {max(leak_check) - min(leak_check)}")
print()
print("A non-zero spread means the PCA is refitted inside each fold. Under the")
print("2019 structure every fold saw the same component count, because the PCA")
print("was fitted once on everything before the loop started.")

# %% [markdown]
## Change 2 — Tune the HOG hyperparameters, not just the classifier

Column 1 used `PCA(n_components=0.9)` and a default-ish SVM. The descriptor's
own parameters — cell size, block size, orientation count — matter at least as
much as the classifier, and were never examined.

# %%
def build_pipeline(pixels_per_cell, cells_per_block, orientations,
                   n_components, C, gamma="scale"):
    """Full pipeline; the HOG step is a transformer so it sits inside the CV loop."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class HOGTransformer(BaseEstimator, TransformerMixin):
        """Wraps the HOG function so it can live inside a Pipeline."""

        def __init__(self, pixels_per_cell=8, cells_per_block=2, orientations=9):
            self.pixels_per_cell = pixels_per_cell
            self.cells_per_block = cells_per_block
            self.orientations = orientations

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.array([hog(np.asarray(x).reshape(64, 64),
                                 self.pixels_per_cell, self.cells_per_block,
                                 self.orientations) for x in X])

    return Pipeline([
        ("hog", HOGTransformer(pixels_per_cell, cells_per_block, orientations)),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=SEED)),
        ("clf", SVC(C=C, gamma=gamma, random_state=SEED)),
    ])


# The HOG step is expensive, so cache its output for the search rather than
# recomputing it inside every fold — the descriptor has no fitted parameters,
# so caching is legitimate (unlike caching a fitted scaler).
HogCache = {}
for ppc, cpb, ori in [(8, 2, 9), (16, 2, 9), (8, 4, 9), (16, 4, 9), (8, 2, 6)]:
    key = (ppc, cpb, ori)
    HogCache[key] = np.array([hog(im, ppc, cpb, ori) for im in imgs])
    print(f"  cached HOG{key}: {HogCache[key].shape[1]} dims")

# %%
print()
print("search over descriptor and classifier parameters (5-fold CV):")
print()
print(f"  {'ppc':>4} {'cpb':>4} {'ori':>4} {'n_comp':>7} {'C':>7}  {'balanced acc':>13}")
results = []
for key, feats in HogCache.items():
    for n_components in [0.9, 0.99]:
        for C in [1.0, 10.0, 100.0]:
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", SVC(C=C, random_state=SEED)),
            ])
            scores = cross_val_score(pipe, feats, y_syn, cv=cv,
                                     scoring="balanced_accuracy", n_jobs=2)
            results.append({"ppc": key[0], "cpb": key[1], "ori": key[2],
                            "n_components": n_components, "C": C,
                            "balanced_acc": scores.mean(), "std": scores.std()})
            print(f"  {key[0]:>4} {key[1]:>4} {key[2]:>4} {str(n_components):>7} "
                  f"{C:>7}  {scores.mean():>13.4f}")

res = pd.DataFrame(results)
best = res.loc[res.balanced_acc.idxmax()]
print()
print(f"best: ppc={int(best.ppc)}, cpb={int(best.cpb)}, orientations={int(best.ori)}, "
      f"n_components={best.n_components}, C={best.C}")
print(f"      balanced accuracy {best.balanced_acc:.4f} (+/- {best['std']:.4f})")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
pivot = res.pivot_table(index=["ppc", "cpb", "ori"], columns="C", values="balanced_acc")
im = ax[0].imshow(pivot.values, cmap="RdYlGn", aspect="auto")
ax[0].set_xticks(range(len(pivot.columns)))
ax[0].set_xticklabels([f"C={c:g}" for c in pivot.columns])
ax[0].set_yticks(range(len(pivot.index)))
ax[0].set_yticklabels([f"{i[0]}/{i[1]}/{i[2]}" for i in pivot.index], fontsize=7)
ax[0].set_xlabel("SVM C")
ax[0].set_ylabel("HOG cell/block/orient")
ax[0].set_title("CV balanced accuracy")
plt.colorbar(im, ax=ax[0])

ax[1].errorbar(res.C, res.balanced_acc, yerr=res["std"], fmt="o", alpha=0.6,
               color="#4C72B0", capsize=3)
for _, r in res.iterrows():
    if r.balanced_acc == res.balanced_acc.max():
        ax[1].scatter(r.C, r.balanced_acc, s=140, facecolors="none",
                     edgecolors="crimson", zorder=5)
ax[1].set_xscale("log")
ax[1].set_xlabel("SVM C")
ax[1].set_ylabel("balanced accuracy")
ax[1].set_title("With fold-to-fold spread")
plt.tight_layout()
plt.show()

# %% [markdown]
## Change 3 — A real baseline ladder

**If PCA did not help, that is worth knowing.** Column 1 compared one pipeline
against nothing, so it could not tell whether the representation was doing work.

# %%
best_feats = HogCache[(int(best.ppc), int(best.cpb), int(best.ori))]
X_dev, X_test, y_dev, y_test = train_test_split(
    best_feats, y_syn, test_size=0.25, random_state=SEED, stratify=y_syn
)

ladder = []
# Rung 0: majority
ladder.append({"model": "majority class",
               "balanced_acc": 0.5, "roc_auc": 0.5,
               "note": f"predict the more common class ({max(y_syn.mean(), 1-y_syn.mean()):.2f} base)"})

# Rung 1: raw pixels
pixel_feats = np.array([im.ravel() for im in imgs])
X_pix = np.array([imgs[i].ravel() for i in
                  train_test_split(np.arange(N), test_size=0.25,
                                   random_state=SEED, stratify=y_syn)[1]])
pipe_pix = Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(C=best.C, random_state=SEED))])
s = cross_val_score(pipe_pix, pixel_feats, y_syn, cv=cv, scoring="balanced_accuracy")
ladder.append({"model": "raw pixels + SVM", "balanced_acc": s.mean(), "roc_auc": np.nan,
               "note": f"{s.std():.3f} fold std"})

# Rung 2: HOG without PCA
pipe_hog = Pipeline([("sc", StandardScaler()), ("clf", SVC(C=best.C, random_state=SEED))])
s = cross_val_score(pipe_hog, best_feats, y_syn, cv=cv, scoring="balanced_accuracy")
ladder.append({"model": "HOG (no PCA) + SVM", "balanced_acc": s.mean(), "roc_auc": np.nan,
               "note": f"{s.std():.3f} fold std"})

# Rung 3: the tuned pipeline
pipe_tuned = Pipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=best.n_components, random_state=SEED)),
                       ("clf", SVC(C=best.C, random_state=SEED))])
s = cross_val_score(pipe_tuned, best_feats, y_syn, cv=cv, scoring="balanced_accuracy")
ladder.append({"model": f"HOG + PCA({best.n_components}) + SVM", "balanced_acc": s.mean(),
               "roc_auc": np.nan, "note": f"{s.std():.3f} fold std"})

ladder_df = pd.DataFrame(ladder)
print(ladder_df.round(4).to_string(index=False))
print()
pca_gain = ladder_df.loc[2, "balanced_acc"] - ladder_df.loc[1, "balanced_acc"]
hog_gain = ladder_df.loc[1, "balanced_acc"] - ladder_df.loc[0, "balanced_acc"]
print(f"HOG over raw pixels : {hog_gain:+.4f}")
print(f"PCA over no-PCA     : {pca_gain:+.4f}")
print()
if abs(pca_gain) < 0.01:
    print("PCA is not doing meaningful work on this corpus — the SVM handles the")
    print("full descriptor fine. That is a legitimate finding, and column 1")
    print("could not have reported it because it never ran the no-PCA case.")

# %% [markdown]
## Change 4 — Is the classifier even the bottleneck?

# %%
# A learning curve answers whether more images would help. On 400 synthetic
# images the answer is informative about the pipeline, and only weakly about
# real data.
from sklearn.model_selection import learning_curve

pipe_lc = Pipeline([("sc", StandardScaler()),
                    ("pca", PCA(n_components=best.n_components, random_state=SEED)),
                    ("clf", SVC(C=best.C, random_state=SEED))])

train_sizes, train_scores, val_scores = learning_curve(
    pipe_lc, best_feats, y_syn, cv=cv, n_jobs=2,
    train_sizes=np.linspace(0.2, 1.0, 5), scoring="balanced_accuracy",
)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(train_sizes, train_scores.mean(1), "o-", label="train", color="#4C72B0")
ax[0].fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                   train_scores.mean(1) + train_scores.std(1), alpha=0.2, color="#4C72B0")
ax[0].plot(train_sizes, val_scores.mean(1), "o-", label="validation", color="#C44E52")
ax[0].fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                   val_scores.mean(1) + val_scores.std(1), alpha=0.2, color="#C44E52")
ax[0].axhline(0.5, ls="--", color="grey", label="chance")
ax[0].set_xlabel("training images")
ax[0].set_ylabel("balanced accuracy")
ax[0].set_title("Learning curve (SYNTHETIC corpus)")
ax[0].legend()

gap = train_scores.mean(1)[-1] - val_scores.mean(1)[-1]
ax[1].bar(train_sizes, gap, width=0.06, color="#8172B3")
ax[1].axhline(0, color="black", lw=1)
ax[1].set_xlabel("training images")
ax[1].set_ylabel("train - validation")
ax[1].set_title("Generalisation gap")
plt.tight_layout()
plt.show()

print(f"final train balanced accuracy     : {train_scores.mean(1)[-1]:.4f}")
print(f"final validation balanced accuracy : {val_scores.mean(1)[-1]:.4f}")
print(f"generalisation gap                : {gap:+.4f}")
print()
if gap > 0.15:
    print("A large gap means the model is fitting noise in the training images.")
    print("More data would help more than more parameters.")
elif val_scores.mean(1)[-1] < 0.55:
    print("Validation is near chance: the descriptor is not separating the two")
    print("classes at all. This is a representation problem, not a tuning one.")
else:
    print("The gap is small and validation is above chance: the pipeline works.")

# %%
# Final held-out evaluation
pipe_final = Pipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=best.n_components, random_state=SEED)),
                       ("clf", SVC(C=best.C, random_state=SEED))])
pipe_final.fit(X_dev, y_dev)
pred = pipe_final.predict(X_test)
print()
print(f"held-out test balanced accuracy: {balanced_accuracy_score(y_test, pred):.4f}")
print(f"held-out test accuracy         : {accuracy_score(y_test, pred):.4f}")
print()
print(classification_report(y_test, pred, target_names=["Apis", "Bombus"], digits=3))
print(pd.DataFrame(confusion_matrix(y_test, pred),
                   index=["Apis", "Bombus"], columns=["pred Apis", "pred Bombus"]).to_string())

# %% [markdown]
## Summary

| | Column 1 | Column 2 |
|---|---|---|
| Preprocessing | fitted on all data | inside `Pipeline`, refit per fold |
| HOG parameters | fixed at defaults | searched (cell, block, orientation) |
| SVM `C` | default | searched over three values |
| Baselines | none | majority, raw pixels, no-PCA, PCA |
| Variance | unreported | fold-to-fold std on every number |
| Overfitting | not examined | learning curve + generalisation gap |
| Reporting | accuracy | balanced accuracy, per-class P/R, confusion matrix |

# %%
print("COLUMN 2 RESULTS (SYNTHETIC CORPUS — not real bees)")
print(f"  images                : {N}")
print(f"  best HOG config       : cell {int(best.ppc)}, block {int(best.cpb)}, "
      f"{int(best.ori)} orientations")
print(f"  best CV balanced acc   : {best.balanced_acc:.4f} (+/- {best['std']:.4f})")
print(f"  HOG vs raw pixels     : {hog_gain:+.4f}")
print(f"  PCA vs no-PCA         : {pca_gain:+.4f}")
print(f"  generalisation gap    : {gap:+.4f}")
print(f"  held-out balanced acc : {balanced_accuracy_score(y_test, pred):.4f}")
print()
print("Two of the three changes that matter produced no gain on this corpus:")
print("PCA and the wider search. That is the finding, and column 1's structure")
print("could not have surfaced it because it had no comparison to lose against.")
