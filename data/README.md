# Datasets

Small datasets are committed so the repository is usable straight after
cloning. Anything larger is fetched on demand:

```bash
python scripts/fetch_data.py --list     # available groups + manual steps
python scripts/fetch_data.py --check    # what is missing
python scripts/fetch_data.py            # download everything available
```

## Committed

| Path | Size | Used by | Source |
|---|--:|---|---|
| `statistics/bank_data.csv` | 86 KB | Regression discontinuity | DataCamp *Which Debts Are Worth the Bank's Effort* |
| `machine-learning/cc_approvals.data` | 32 KB | Credit card approvals | UCI Credit Approval (CC BY 4.0) |
| `machine-learning/transfusion.data` | 12 KB | TPOT blood donations | UCI Blood Transfusion Service Center (CC BY 4.0) |
| `nlp/nytkids_yearly.csv` | 28 KB | NYSIIS notebook | NYT bestsellers, 2008-2017 |
| `nlp/babynames_nysiis.csv` | 296 KB | NYSIIS notebook | SSA baby names, NYSIIS-encoded |
| `nlp/project_image.png` | 84 KB | NYSIIS notebook | Project diagram |
| `computer-vision/naive-bees/labels.csv` | 4 KB | Naive Bees | DataCamp *Naive Bees* |
| `computer-vision/asl-letters/sign_language.py` | 5 KB | ASL CNN | Rewritten loader (see below) |
| `sql/international_debt.zip` | 49 KB | SQL notebook | World Bank debt statistics (CC BY 4.0) |

## Fetched automatically

Every URL in `scripts/fetch_data.py` is verified to return HTTP 200.

| Key | Path | Size |
|---|---|--:|
| `song-genres` | `machine-learning/fma-rock-vs-hiphop.csv` | 3.3 MB |
| `song-genres` | `machine-learning/echonest-metrics.json` | 2.1 MB |

## Manual downloads

Three datasets have no stable direct link. `fetch_data.py --only <key>` prints
the steps.

| Key | Why not automated |
|---|---|
| `lda` | The NIPS corpus is ~200 MB, over GitHub's raw-file limit. Get it from Kaggle: `benhamner/nips-papers` |
| `asl-letters` | The Udacity S3 bucket (`udacitysaints`) no longer exists — `NoSuchBucket` |
| `bees-images` | No reliable raw mirror of the DataCamp image archive |

**Until you supply any of these, the corresponding notebook generates a
clearly-labelled synthetic stand-in**, prints a warning, and titles every figure
`SYNTHETIC DATA`. A perfect score on generated shapes is not evidence, and the
notebooks say so rather than letting the number stand.

## Removed, and why

| Removed | Reason |
|---|---|
| `papers.csv.gz.part_aa` ... `.part_ah` (**110 MB**) | A multi-part download that was never reassembled. The notebook could not run, and the fragments were committed anyway. Replaced by a documented manual step plus a synthetic fallback |
| `bee_1.jpg`, `bee_2.jpg`, `bee_3.jpg`, `bee_12.jpg` | **Irrecoverably corrupt.** Every byte `>= 0x80` had been replaced with a UTF-8 replacement character, so the JPEG markers and entropy-coded data were both destroyed. `PIL.Image.open` raised `UnidentifiedImageError` on all four, and the damage cannot be undone — see `BUGFIXES.md` |
| `A.zip`, `B.zip`, `C.zip` (**26 MB**) | ASL photographs, replaced by the manual-download path above |
| `*.rar`, duplicate `.m` files | Extracted and de-duplicated into the MATLAB repository |

## Rewritten

`computer-vision/asl-letters/sign_language.py` is new. The original loader sat
in a `datasets/` folder that was not a Python package, was imported with
`from datasets import sign_language` (which also collides with the unrelated
PyPI `datasets` distribution), and used `keras.utils.np_utils`, removed in
Keras 3. The replacement depends only on NumPy, Pillow and the standard library,
so it works under any Keras version.

## Licences

Datasets remain under their original licences:

- **UCI Credit Approval**, **UCI Blood Transfusion** - CC BY 4.0
- **World Bank International Debt Statistics** - CC BY 4.0
- **SSA baby names** - US Social Security Administration, public domain
- **FMA / echonest audio features** - Free Music Archive, CC BY-NC-SA
- **NIPS paper metadata** - Kaggle `benhamner/nips-papers`
- **Naive Bees, bank debt, NYT bestsellers, ASL letters** - DataCamp course
  datasets, used under their course terms
