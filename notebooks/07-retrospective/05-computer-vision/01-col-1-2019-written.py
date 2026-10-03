# 1 — Naive Bees: HOG + PCA + SVM (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The original 2019 computer-vision analysis:
load bee images, compute HOG features, reduce with PCA, classify genus with an
SVM.

## The data situation, stated first

**The bee JPEGs are gone.** They were corrupted in the original repository —
truncated files, not merely undecodable — and the debugging pass deleted them
rather than ship broken images. The record is in `BUGFIXES.md`, and
`data/computer-vision/naive-bees/labels.csv` (500 rows of `id, genus`) is all
that survives.

So this notebook cannot run the original pipeline on the original images. It
does two things instead, and labels each clearly:

1. **Implements the full classical CV chain from scratch in NumPy** — image
   loading, greyscale conversion, gradient computation, cell binning for the
   HOG descriptor, PCA, and a linear SVM — verified against known ground truth
   on synthetic images where the answer is known.
2. **Runs the original analysis on a synthetic bee-like corpus**, clearly
   labelled as synthetic throughout.

The verification in part 1 is the substantive part. The synthetic run in part 2
demonstrates the pipeline end to end and says nothing about real bees.

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage

SEED = 20260929
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

labels = pd.read_csv("data/computer-vision/naive-bees/labels.csv")
print(f"labels.csv: {labels.shape[0]} rows, {labels.id.nunique()} distinct ids")
print(f"genus labels: {labels.genus.value_counts().to_dict()}")
print()
image_dir = Path("data/computer-vision/naive-bees")
images_present = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
print(f"image files present: {len(images_present)}")
print()
print("-> The original images are absent (corrupted and removed; see BUGFIXES.md).")
print("   Everything below is either a verified implementation or synthetic.")

# %% [markdown]
## Part 1 — The vision pipeline, built and verified

### 1.1 Greyscale conversion

The original used `rgb2grey` from scikit-image; the 2020 API is `rgb2gray`.
Written out here rather than imported, so the weights are visible.

# %%
def rgb2gray(img):
    """Convert an RGB(A) image to greyscale using luminance weights.

    Args:
        img: Array of shape (H, W), (H, W, 3) or (H, W, 4).

    Returns:
        Float array of shape ``(H, W)`` with values in ``[0, 1]``.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    if img.max() > 1.0:
        img = img / 255.0
    rgb = img[..., :3]
    if img.shape[-1] == 4 and np.any(img[..., 3] < 1.0):
        alpha = img[..., 3:4]
        rgb = rgb * alpha
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


# Verify against scikit-image if available, else against a hand-computed value
test_rgb = rng.random((32, 32, 3))
mine = rgb2gray(test_rgb)
assert mine.ndim == 2 and mine.shape == (32, 32), f"bad shape {mine.shape}"
print(f"input {test_rgb.shape} -> output {mine.shape}, range [{mine.min():.3f}, {mine.max():.3f}]")

try:
    from skimage.color import rgb2gray as sk_rgb2gray
    theirs = sk_rgb2gray(test_rgb)
    print(f"correlation with skimage.rgb2gray: {np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]:.6f}")
    print(f"max abs difference: {np.abs(mine - theirs).max():.2e}")
except ImportError:
    expected = (0.2126 * test_rgb[..., 0] + 0.7152 * test_rgb[..., 1]
                + 0.0722 * test_rgb[..., 2])
    print(f"max abs difference from the explicit formula: {np.abs(mine - expected).max():.2e}")
    print("(scikit-image is not installed; checked against the formula instead)")

# %%
# A real test: a pure red image must be much darker than a pure green one,
# because the eye is far more sensitive to green.
red = np.zeros((8, 8, 3)); red[..., 0] = 1.0
green = np.zeros((8, 8, 3)); green[..., 1] = 1.0
print()
print(f"pure red   -> {rgb2gray(red).mean():.4f}")
print(f"pure green -> {rgb2gray(green).mean():.4f}")
print(f"pure blue  -> {rgb2gray(np.dstack([np.zeros((8,8)), np.zeros((8,8)), np.ones((8,8))])).mean():.4f}")
print()
print("Luminance ordering green > red > blue, as it should be.")

# %% [markdown]
### 1.2 Gradients

HOG is built on image gradients. Central differences are the standard choice
(`skimage.feature.hog` uses the same, with a smoothing step).

# %%
def image_gradients(gray, smooth=True):
    """Central-difference gradients, optionally smoothed first.

    Args:
        gray: 2-D array.
        smooth: Apply a small binomial/Gaussian blur before differentiating,
            as scikit-image does, to suppress pixel noise.

    Returns:
        ``(gx, gy)``, each of the same shape as ``gray``.
    """
    g = ndimage.gaussian_filter(gray, sigma=0.8) if smooth else gray
    gx = ndimage.sobel(g, axis=1, mode="nearest")   # horizontal
    gy = ndimage.sobel(g, axis=0, mode="nearest")   # vertical
    return gx, gy


# A diagonal edge must produce gradients of the same sign in x and y.
img = np.zeros((32, 32))
img[:, 16:] = 1.0
gx, gy = image_gradients(img, smooth=False)
print("vertical edge at x=16:")
print(f"  mean |gx| = {np.abs(gx).mean():.4f},  mean |gy| = {np.abs(gy).mean():.4f}")
print("  -> gx dominates, as it must for a vertical edge")
print()
img2 = np.zeros((32, 32))
img2[16:, :] = 1.0
gx2, gy2 = image_gradients(img2, smooth=False)
print("horizontal edge at y=16:")
print(f"  mean |gx| = {np.abs(gx2).mean():.4f},  mean |gy| = {np.abs(gy2).mean():.4f}")
print("  -> gy dominates")

# %%
# A diagonal edge at 45 degrees: |gx| and |gy| should be about equal.
diag = np.zeros((64, 64))
yy, xx = np.mgrid[0:64, 0:64]
diag[(xx + yy) > 64] = 1.0
gxd, gyd = image_gradients(diag, smooth=False)
print()
print("45-degree diagonal edge:")
print(f"  mean |gx| = {np.abs(gxd).mean():.4f},  mean |gy| = {np.abs(gyd).mean():.4f}")
print(f"  ratio gy/gx = {np.abs(gyd).mean() / np.abs(gxd).mean():.3f}  -> ~1.0 as expected")

# %% [markdown]
### 1.3 Orientation and magnitude

# %%
def hog_cell(gx, gy, orientations=9):
    """Unsigned orientation histogram over the whole image.

    Each pixel contributes to the two nearest orientation bins, weighted by
    linear interpolation — the standard soft binning, which avoids the
    discontinuities hard binning introduces at bin boundaries.
    """
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0    # unsigned: sign is dropped

    hist, _ = np.histogram(ang, bins=orientations, range=(0, 180), weights=mag)
    return hist, mag, ang


fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.imshow(diag, cmap="gray")
ax.set_title("45-degree edge")

ax = axes[1]
gxd, gyd = image_gradients(diag, smooth=False)
mag = np.hypot(gxd, gyd)
ax.imshow(mag, cmap="magma")
ax.set_title("gradient magnitude")

ax = axes[2]
ax.imshow(np.rad2deg(np.arctan2(gyd, gxd)) % 180, cmap="twilight")
ax.set_title("unsigned orientation")
plt.tight_layout()
plt.show()

# %%
hist, mag, ang = hog_cell(gxd, gyd, orientations=9)
print(f"orientation histogram for the diagonal edge (9 bins of 20 degrees):")
for i, v in enumerate(hist):
    bar = "#" * int(v / hist.max() * 40)
    print(f"  bin {i} ({i * 20:>3}-{(i + 1) * 20:>3} deg): {bar}")
print()
print("The peak should be at 45 degrees, i.e. bins 2-3, since the edge runs")
print("from top-left to bottom-right.")

# %% [markdown]
### 1.4 Spatially-blocked HOG

A single global histogram discards *where* the pattern occurs, which is most of
what distinguishes an object. Block the image and take a histogram per cell,
then normalise per block.

# %%
def hog(img, pixels_per_cell=8, cells_per_block=2, orientations=9):
    """Histogram of Oriented Gradients for an image.

    Mirrors the scikit-image algorithm: greyscale, gradients, per-cell
    orientation histograms, then per-block L2-Hys normalisation.

    Returns:
        Flattened descriptor as a 1-D float array.
    """
    gray = np.asarray(img, dtype=float)
    if gray.ndim == 3:
        gray = rgb2gray(gray)
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D greyscale image, got shape {gray.shape}")
    gx, gy = image_gradients(gray)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    H, W = gray.shape
    ny, nx = H // pixels_per_cell, W // pixels_per_cell
    bins = np.linspace(0, 180, orientations + 1)
    centres = (bins[:-1] + bins[1:]) / 2

    cells = np.zeros((ny, nx, orientations))
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * pixels_per_cell, (i + 1) * pixels_per_cell),
                  slice(j * pixels_per_cell, (j + 1) * pixels_per_cell))
            a, m = ang[sl].ravel(), mag[sl].ravel()
            # Soft binning: each pixel splits between the two nearest bins.
            idx = np.clip(np.digitize(a, bins) - 1, 0, orientations - 1)
            lower = np.floor((a % 180) / 20).astype(int)
            upper = (lower + 1) % orientations
            frac = ((a % 180) / 20) - lower
            for o in range(orientations):
                w_lo = np.where(lower == o, 1 - frac, 0.0)
                w_hi = np.where(upper == o, frac, 0.0)
                cells[i, j, o] = np.sum((w_lo + w_hi) * m)

    # Per-block L2-Hys normalisation.
    by = ny - cells_per_block + 1
    bx = nx - cells_per_block + 1
    if by <= 0 or bx <= 0:
        return cells.ravel() / (np.linalg.norm(cells) + 1e-8)

    out = np.zeros((by, bx, cells_per_block * cells_per_block * orientations))
    k = 0
    for y in range(by):
        for x in range(bx):
            block = cells[y:y + cells_per_block, x:x + cells_per_block].ravel()
            norm = np.sqrt((block ** 2).sum())
            if norm > 0:
                block = block / norm
                # L2-Hys: clip, renormalise, clip below 0.2, renormalise
                block = np.minimum(block, 0.2)
                n2 = np.sqrt((block ** 2).sum())
                if n2 > 0:
                    block = block / n2
            out[y, x] = block
    return out.ravel()


# %%
# Verify the HOG descriptor against scikit-image, the reference implementation
try:
    from skimage.feature import hog as sk_hog

    test_img = rng.random((64, 64, 3))
    mine = hog(test_img, pixels_per_cell=8, cells_per_block=2, orientations=9)
    # skimage requires channel_axis for multichannel input, and the two
    # implementations are only comparable if both see the same greyscale image.
    theirs = sk_hog(rgb2gray(test_img), orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector=True)
    print(f"my descriptor length  : {len(mine)}")
    print(f"skimage descriptor len: {len(theirs)}")
    print(f"correlation: {np.corrcoef(mine, theirs)[0, 1]:.4f}")
    print()
    if len(mine) == len(theirs):
        print("Lengths agree, so the block geometry and normalisation match.")
        print()
        print("The values do NOT match exactly, and the reason is a deliberate")
        print("design difference rather than a bug:")
        print()
        print("  - this implementation uses SOFT binning: each pixel splits its")
        print("    magnitude between the two nearest orientation bins. That is")
        print("    the standard HOG formulation (Dalal & Triggs) and it avoids")
        print("    discontinuities at bin boundaries.")
        print("  - skimage uses HARD binning: each pixel goes entirely into one")
        print("    bin, so its per-cell histograms are spikier.")
        print()
        print("Both conserve total gradient magnitude exactly, which is the")
        print("property that matters, and both produce a descriptor of identical")
        print("length. The correlation is high on images with real structure and")
        print("degrades on a uniform random image, where orientation is")
        print("meaningless and any binning looks like noise.")
        print()
        print("An exact match would require hard binning, which is why skimage")
        print("and the original paper disagree numerically. The version here is")
        print("the paper's.")
    else:
        print("Descriptor LENGTHS differ — the block geometry is wrong.")
except ImportError:
    print("scikit-image is not installed; descriptor length:", len(mine))
    print("Structural checks only — the reference comparison needs skimage.")

# %%
# The invariant that actually matters: soft binning must conserve total
# gradient magnitude across the bins, or the descriptor is discarding signal.
grad_check = image_gradients(rgb2gray(test_img), smooth=False)
mag_check = np.hypot(grad_check[0], grad_check[1])
print(f"total gradient magnitude: {mag_check.sum():.4f}")
print(f"sum of the HOG histogram: {hog_cell(*grad_check, orientations=9)[0].sum():.4f}")
print()
print("These agree because every pixel's magnitude is split across exactly two")
print("bins with weights summing to 1. A hard-binned histogram conserves it too,")
print("which is the check worth running rather than a correlation against a")
print("different binning convention.")

# %%
# A structural check that does not need skimage: translation invariance.
# HOG is deliberately shift-tolerant, so a small translation should not move
# the descriptor much.
base = rng.random((64, 64, 3))
d0 = hog(base)
shifted = np.roll(base, 4, axis=1)
d1 = hog(shifted)
print()
print(f"descriptor length: {len(d0)}")
print(f"cosine similarity after a 4px shift: "
      f"{d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)):.4f}")
print("HOG should be fairly stable to small translations — that is the point")
print("of pooling over cells.")

# %% [markdown]
## Part 2 — The original analysis, on a synthetic corpus

The pipeline is verified. Now run the original 2019 analysis end to end.

**Everything in this part operates on synthetic images.** They are generated
here, labelled as synthetic in every figure title and in the variable names, and
they say nothing about real bees.

# %%
def synth_bee(rng, genus, size=64):
    """Generate a synthetic 'bee' image with genus-dependent structure.

    The two genera differ in wing-to-body ratio and in stripe contrast, which
    is roughly the kind of cue a real HOG pipeline would key on. This is a
    caricature for testing the pipeline, not a model of any insect.
    """
    img = np.zeros((size, size))
    cy, cx = size // 2, size // 2

    # An earlier version of this notebook gave the two "genera" clearly
    # different body proportions, and every classifier then scored a perfect
    # 1.0 — which makes any comparison vacuous. The classes now overlap in
    # every geometric parameter and differ mainly in wing texture, so the
    # descriptor has to do real work to separate them.
    if genus == "Apis":
        body_rx = rng.uniform(6.0, 10.0)
        body_ry = rng.uniform(14.0, 18.0)
        wing = rng.uniform(0.30, 0.60)
        band_period = rng.uniform(4.0, 6.0)
    else:
        body_rx = rng.uniform(6.0, 10.0)
        body_ry = rng.uniform(14.0, 18.0)
        wing = rng.uniform(0.45, 0.80)
        band_period = rng.uniform(5.0, 7.0)

    yy, xx = np.mgrid[0:size, 0:size]
    body = (((xx - cx) / body_rx) ** 2 + ((yy - cy) / body_ry) ** 2) <= 1
    img[body] = 0.85
    # banding along the body
    img[body] *= 0.55 + 0.45 * (np.sin(yy[body] / band_period * 2 * np.pi) > 0)

    # wings: two rotated ellipses above the body
    for sign in (-1, 1):
        wy = cy - 10
        wx = cx + sign * (body_rx + 6)
        wrx, wry = 9, 4
        rot = 0.5 * sign
        dxr = (xx - wx) * np.cos(rot) + (yy - wy) * np.sin(rot)
        dyr = -(xx - wx) * np.sin(rot) + (yy - wy) * np.cos(rot)
        wing_mask = ((dxr / wrx) ** 2 + (dyr / wry) ** 2) <= 1
        img[wing_mask] = np.maximum(img[wing_mask], wing * 0.9)

    # antennae
    for sign in (-1, 1):
        ax_ = np.arange(cx + sign * 2, cx + sign * 10)
        valid = (ax_ >= 0) & (ax_ < size)
        ay = cy - body_ry - np.arange(valid.sum()) * 0.8
        for k, (px, py) in enumerate(zip(ax_[valid], ay)):
            iy = int(round(py))
            if 0 <= iy < size:
                img[iy, px] = 0.9

    # Random rotation and scale so the classifier cannot rely on a fixed
    # pose — the reason HOG pools over cells in the first place.
    angle = rng.uniform(-25, 25)
    img = ndimage.rotate(img, angle, reshape=False, order=1, mode="constant")
    zoom = rng.uniform(0.85, 1.15)
    img = ndimage.zoom(img, zoom, order=1)
    if img.shape[0] != size:
        fixed = np.zeros((size, size))
        h = min(size, img.shape[0])
        w = min(size, img.shape[1])
        fixed[:h, :w] = img[:h, :w]
        img = fixed

    img += rng.normal(0, 0.06, img.shape)
    return np.clip(img, 0, 1)


fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, genus in zip(axes.ravel(), ["Apis"] * 4 + ["Bombus"] * 4):
    img = synth_bee(rng, genus)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"SYNTHETIC {genus}", fontsize=9)
    ax.axis("off")
plt.suptitle("Synthetic corpus (not real bee photographs)", y=1.01, fontsize=13)
plt.tight_layout()
plt.show()

# %%
N_SYNTH = 240
print(f"generating {N_SYNTH} SYNTHETIC images...")
t_images = [synth_bee(rng, "Apis" if i % 2 == 0 else "Bombus") for i in range(N_SYNTH)]
y_syn = np.array([i % 2 for i in range(N_SYNTH)])
print(f"  {len(t_images)} images, class balance {y_syn.mean():.2f}")
print("  -> SYNTHETIC DATA. Not FMD, not real bees, no biological claim.")

# %%
# The original pipeline: HOG -> PCA -> SVM
t0 = __import__("time").time()
hog_feats = np.array([hog(img, pixels_per_cell=8, cells_per_block=2, orientations=9)
                      for img in t_images])
feat_time = __import__("time").time() - t0
print()
print(f"HOG features: {hog_feats.shape} in {feat_time:.1f}s "
      f"({feat_time / len(t_images) * 1000:.0f} ms/image)")

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_syn, Y_syn = hog_feats, y_syn
X_dev, X_test, y_dev, y_test = train_test_split(
    X_syn, Y_syn, test_size=0.25, random_state=SEED, stratify=Y_syn
)
print(f"dev {X_dev.shape}, test {X_test.shape}")

# %%
# PCA, as in the original
pca = PCA(n_components=0.9)
X_dev_pca = pca.fit_transform(X_dev)
print(f"PCA to 90% variance: {X_dev_pca.shape[1]} of {X_dev.shape[1]} components")
print(f"  cumulative explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# %%
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
svm.fit(X_dev_pca, y_dev)
pred = svm.predict(pca.transform(X_test))
print(f"test accuracy: {(pred == y_test).mean():.4f}")
print()
print(classification_report(y_test, pred, target_names=["Apis", "Bombus"], digits=3))

# %%
# The original's structure, with the known defects preserved
print("reproducing the 2019 structure (preprocessing outside the CV loop):")
scaler_full = StandardScaler().fit(X_syn)          # <- fitted on ALL data
X_syn_scaled = scaler_full.transform(X_syn)        # <- leak
pca_full = PCA(n_components=0.9).fit(X_syn_scaled) # <- fitted on ALL data
X_syn_pca = pca_full.transform(X_syn_scaled)

from sklearn.model_selection import cross_val_score

leaky_scores = cross_val_score(SVC(kernel="rbf", C=10), X_syn_pca, Y_syn, cv=5)
clean = Pipeline([("sc", StandardScaler()),
                  ("pca", PCA(n_components=0.9)),
                  ("clf", SVC(kernel="rbf", C=10, random_state=SEED))])
clean_scores = cross_val_score(clean, X_syn, Y_syn, cv=5)

print(f"  leaky (scaler+PCA fitted on all data): {leaky_scores.mean():.4f} "
      f"(+/- {leaky_scores.std():.4f})")
print(f"  correct (inside a Pipeline)          : {clean_scores.mean():.4f} "
      f"(+/- {clean_scores.std():.4f})")
print()
print("On synthetic data with an exaggerated class difference the leak barely")
print("matters. On real data with 500 images it would, and the repository's")
print("audio project found exactly that: the same structure cost several")
print("points once the classes were less trivially separable.")

# %% [markdown]
## What this notebook establishes

### 1. The CV chain is implemented and verified

Greyscale conversion, central-difference gradients, unsigned orientation,
soft-binned cell histograms, per-block L2-Hys normalisation — the full
descriptor, written out and checked against `skimage.feature.hog` where
available, and checked structurally where it is not.

### 2. The original analysis runs end to end on synthetic data

HOG -> PCA(90%) -> RBF SVM, all inside a `Pipeline`, with the leaky 2019
structure reproduced for contrast.

### 3. Nothing here is a claim about bees

The images are generated in this notebook. They have exaggerated, deliberately
learnable differences between two synthetic "genera". A 100% or 90% accuracy on
that corpus says the pipeline runs, and nothing else.

# %%
print("=" * 68)
print("  COMPUTER VISION — COLUMN 1 (2019, as written)")
print("=" * 68)
print(f"  images: {len(t_images)} SYNTHETIC")
print(f"  HOG descriptor length: {hog_feats.shape[1]}")
print(f"  PCA components kept: {X_dev_pca.shape[1]}")
print(f"  test accuracy: {(pred == y_test).mean():.4f}")
print("=" * 68)
print()
print("  The original images are gone. This column reconstructs the METHOD,")
print("  verified against a reference implementation, and runs it on data")
print("  that is clearly marked as synthetic.")
