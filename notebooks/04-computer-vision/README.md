# Computer vision

Three image projects, moving from pixel manipulation to learned features.

---

## 1. Naive Bees — image manipulation

**File:** [`01-naive-bees-image-manipulation.ipynb`](01-naive-bees-image-manipulation.ipynb)
**Status:** ✅ runs
**Competencies:** image processing ●● · statistical visualisation ●·

Foundations: cropping, rotation, flipping, resizing, greyscale conversion and
contrast stretching with PIL, plus per-channel kernel density estimates used to
compare the colour distributions of a honey bee against a bumblebee.

The density plots are the useful part — they make "these look different"
quantitative, and they motivate dropping colour entirely in the next project.

Finishes by batching the transforms into a reusable `process_image` pipeline
over a list of files.

### Reproducing

```bash
pip install numpy pandas matplotlib Pillow
jupyter notebook 01-naive-bees-image-manipulation.ipynb
```

---

## 2. Naive Bees — HOG, PCA and SVM

**File:** [`02-naive-bees-hog-pca-svm.ipynb`](02-naive-bees-hog-pca-svm.ipynb)
**Status:** ✅ runs
**Competencies:** feature engineering ●● · dimensionality reduction ●· image signal processing ●●

**The strongest computer-vision project here.** Classifies bee genus (Apis vs
Bombus) from photographs.

**Pipeline:** load images → convert to greyscale → compute **Histogram of
Oriented Gradients** with `pixels_per_cell=(16, 16)` and `block_norm='L2-Hys'`
→ concatenate with flattened colour channels → standardise → PCA to 500
components → SVM with `probability=True` → ROC curve and AUC.

**Why HOG is signal processing.** It convolves the image with gradient
operators, pools the results over spatial cells, and block-normalises. That is
a real image-processing pipeline, and it is the best evidence of that
competency in the repository.

**Why ROC/AUC rather than accuracy.** For a two-class problem, a single
accuracy number hides the operating point. The ROC curve shows the full
trade-off across thresholds.

### Reproducing

```bash
pip install numpy pandas scikit-learn scikit-image matplotlib Pillow
python ../../scripts/fetch_data.py --only bees-images   # optional, full set
jupyter notebook 02-naive-bees-hog-pca-svm.ipynb
```

The four `bee_*.jpg` files that were committed were **irrecoverably corrupt** —
every byte `>= 0x80` had been replaced with a UTF-8 replacement character, so
they decoded to nothing. They are deleted. Without the real set, the notebook
generates a clearly-labelled synthetic stand-in.

One caveat worth stating plainly: on the synthetic stand-in the pipeline scores
**1.000 accuracy and 1.000 AUC**, which is meaningless — generated shapes are
trivially separable by HOG. The real photographs land in the low-to-mid 0.9s.
The notebook says this in its own output.

Also fixed: `skimage.color.rgb2grey` → `rgb2gray` (renamed in 0.19, removed in
0.23), and a hard-coded `datasets/2194.jpg` path that could not work from a
clone.

---

## 3. ASL recognition with a CNN

**File:** [`03-asl-recognition-cnn.ipynb`](03-asl-recognition-cnn.ipynb)
**Status:** ✅ runs end to end
**Competencies:** neural network design ●● · error analysis ●●

### What was intended

A small convolutional network for ASL letter classification: two conv + max-pool
blocks, flatten, dense softmax over three classes. One-hot labels, RMSProp,
categorical cross-entropy, a validation split, and finally a pass over the
misclassified images.

### What was fixed

| Problem | Fix |
|---|---|
| `tf.set_random_seed(2)` | TensorFlow 1 API, removed in TF2 → `tf.random.set_seed` |
| `from keras.utils import np_utils` | `np_utils` no longer exists in Keras 3 → one-hot via `np.eye(n)[y]` |
| `from datasets import sign_language` | The folder is not a package and the name collides with the PyPI `datasets` distribution → imported by explicit file path |
| Final cell was bare prose | A `SyntaxError`; the loader was rewritten as a documented module |
| `load_data(..., size=2000)` | Meant *2000 samples*; after the loader was given a proper signature the same call asked for 2000×2000-pixel images — 48 MB each, and the kernel was OOM-killed (exit 137) before printing anything. Both arguments are now passed by name |
| `Conv2D(..., input_shape=...)` in `add()` | Old functional-Keras style → an explicit `Input` layer |
| A 20-inch-wide inline figure | Renders to a very large PNG; also unreadable → 4×3 grid at 8 inches |

### The error analysis now runs as code

The original's closing cell was a *note*: the author had noticed the
misclassified images looked obviously different to a human, concluded the model
was too simple, and proposed augmentation. That is sound reasoning — look at
your failures, form a hypothesis, propose the intervention.

It was a note rather than a cell, though, so the claim could never be checked.
It is now executable code that renders the misclassified images.

### Reproducing

```bash
pip install tensorflow numpy matplotlib Pillow
python ../../scripts/fetch_data.py --only asl-letters
```

The Udacity S3 bucket that hosted the photographs no longer exists, so until
you supply a replacement the notebook generates **clearly-labelled synthetic
letters** and prints a warning. Any accuracy on those describes drawn shapes,
not photographs.
