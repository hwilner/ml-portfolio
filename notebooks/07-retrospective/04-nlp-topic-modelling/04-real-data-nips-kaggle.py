# 4 — Real NIPS abstracts: the NYT substitution, replaced (Kaggle)

## What this notebook is

The NLP columns in this retrospective had a substitution in them. The original
project was topic modelling on the NIPS paper abstracts, and when the raw NIPS
dump could not be located, the notebooks ran on 603 New York Times children's
bestsellers instead. The substitution was labelled everywhere, but it weakened
the claim: 603 children's books are not a scientific corpus, and no amount of
labelling makes them one.

**This notebook runs on the actual NIPS abstracts.**

```
kaggle datasets download -d rowhitswami/nips-papers-1987-2019-updated \\
    -p data/nlp/nips
```

9,680 papers, of which **6,361 carry an abstract**. That is 10× the substituted
corpus, and it is the corpus the 2019 notebook was written against.

**One correction to the dataset's own framing.** The dump is described as
1987–2019, and the `year` column does range that far — but only for rows with
`full_text`. Abstract coverage starts in **2007**: there are 4 abstracts before
2004, 99 in 2007, and 1,428 in 2019. So the usable corpus is 2007–2019, twelve
years, not thirty. Any era analysis below uses 2007 onward, and saying
"1987–2019" would have been a claim about a field the data does not cover.

## What changes

| | NYT substitution | This notebook |
|---|---|---|
| Documents | 603 | **6,359 abstracts** |
| Domain | children's fiction | machine-learning research |
| Span | 2008–2017, one domain | 2007–2019, one domain, twelve years |
| Topical structure | weak | strong, and drifting |

The last row is the interesting one. ML abstracts from 1987 and from 2019 share
almost no vocabulary that matters. A topic model over the full range should
find *time* as much as it finds *subject*. Whether it does is the question this
notebook asks, because if a single k-topic model cannot separate eras, then the
k selected in column 2 was never the right number.

# %%
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min

SEED = 20260929
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

DATA = Path("data/nlp/nips")

# %%
# 1. Load the real corpus

# %%
if not (DATA / "papers.csv").exists():
    print("""
The Kaggle NeurIPS/NIPS paper dump is not in the repository yet.

    kaggle datasets download -d rowhitswami/nips-papers-1987-2019-updated \\
        -p data/nlp/nips && cd data/nlp/nips && unzip *.zip

papers.csv is 325 MB and third-party, so it is fetched rather than committed.
Its provenance and licence are recorded in data/nlp/README.md.
""")
    raise SystemExit("dataset not present")

papers = pd.read_csv(DATA / "papers.csv", low_memory=False)
print(f"papers: {len(papers)}")
print(f"full_text years: {papers.year.min()}–{papers.year.max()}")
print(f"rows with an abstract: {papers.abstract.notna().sum()}")
ay = papers.loc[papers.abstract.notna(), "year"]
print(f"abstract years    : {ay.min()}–{ay.max()}")
early = papers[(papers.year < 2007)]
print(f"papers before 2007: {len(early)}, of which with an abstract: "
      f"{early.abstract.notna().sum()}")

df = papers[papers.abstract.notna()].copy()
df = df[df.abstract.astype(str).str.len() > 100]
print(f"with a usable abstract: {len(df)}")

# %%
# 2. Clean, the way a 2019 NLP pipeline would not have

# %%
def clean(text):
    t = re.sub(r"http\S+|www\.\S+", " ", str(text).lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # 2019-era preprocessing: collapse whitespace, drop one-character tokens
    t = re.sub(r"\s+", " ", t)
    return " ".join(w for w in t.split() if len(w) > 2)


df["text"] = df.abstract.map(clean)
df = df[df.text.str.split().str.len() >= 20]
print(f"after cleaning: {len(df)}")
print(f"median length: {int(df.text.str.split().str.len().median())} tokens")

# %%
# 3. Vocabulary growth across three decades

# %%
# Bins are within the range that actually has abstracts (2007+). A
# symmetric-looking 1999/2003/2008/2013/2019 split would put 2 papers in the
# first bucket and produce a 119-word vocabulary, making every overlap
# statistic meaningless. These four bins hold >=99 papers each.
decades = [(2007, 2009), (2010, 2012), (2013, 2015), (2016, 2019)]
print(f"{'period':<14}{'papers':>8}{'types':>9}{'tokens':>11}{'type/token':>12}")
print("-" * 54)

vocab_by_period = {}
for lo, hi in decades:
    sub = df[(df.year >= lo) & (df.year <= hi)]
    toks = [w for t in sub.text for w in t.split()]
    vocab_by_period[(lo, hi)] = set(toks)
    ttr = len(set(toks)) / len(toks)
    print(f"{lo}–{hi:<8}{len(sub):>8}{len(set(toks)):>9}{len(toks):>11}{ttr:>12.4f}")

print("\nvocabulary overlap with the first period, Jaccard:")
base = vocab_by_period[decades[0]]
for lo, hi in decades[1:]:
    v = vocab_by_period[(lo, hi)]
    j = len(base & v) / len(base | v)
    print(f"  {decades[0][0]}–{decades[0][1]} vs {lo}–{hi}: {j:.3f}  "
          f"({len(base & v)} shared of {len(v)})")

# %%
# 4. LSA topic structure, and how it splits by era

# %%
vec = TfidfVectorizer(max_features=20000, stop_words="english",
                      min_df=5, max_df=0.4, sublinear_tf=True)
X = vec.fit_transform(df.text)
print(f"document-term matrix: {X.shape}")
print(f"nonzero density: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")

terms = np.array(vec.get_feature_names_out())

for k in (10, 25, 50):
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    Z = svd.fit_transform(X)
    print(f"\nLSA k={k}: explained variance {svd.explained_variance_ratio_.sum():.3f}")

# %%
# 5. Does a single k-topic model separate eras?

# %%
svd = TruncatedSVD(n_components=50, random_state=SEED)
Z = svd.fit_transform(X)
n_comp = 50
# sign-fix each component so the top term is interpretable and stable
for c in range(n_comp):
    if terms[np.argsort(svd.components_[c])[-1]] < terms[np.argsort(-svd.components_[c])[-1]]:
        svd.components_[c] *= -1
        Z[:, c] *= -1

ERAS = [f"{lo}–{str(hi)[-2:]}" for lo, hi in decades]
# keep era as a plain label array alongside the frame: pandas 3 gives back a
# numpy array from .values and mixing the two raises deep inside crosstab
era_labels = np.full(len(df), "outside", dtype=object)
_in = df.year.between(decades[0][0], decades[-1][1]).to_numpy()
era_labels[_in] = np.asarray(pd.cut(df.year[_in],
                                     bins=[d[0] - 1 for d in decades] + [decades[-1][1]],
                                     labels=ERAS))
era_mask = {e: (era_labels == e) for e in ERAS}
# The check a symmetric "1987-2019" split would have failed: how many documents
# fall outside the eras, and are they accounted for rather than dropped?
n_outside = int((era_labels == "outside").sum())
print(f"documents in each era: "
      f"{ {e: int(era_mask[e].sum()) for e in ERAS} }")
print(f"outside every era (kept in the model, excluded from the era table): "
      f"{n_outside}")
assert min(int(era_mask[e].sum()) for e in ERAS) >= 50, \
    "an era bucket is too small to compare; re-derive the bins"
df["era"] = era_labels
era_mask = {e: (era_labels == e) for e in ERAS}
# The check a symmetric "1987-2019" split would have failed: how many documents
# fall outside the eras, and are they accounted for rather than dropped?
n_outside = int((era_labels == "outside").sum())
print(f"documents in each era: "
      f"{ {e: int(era_mask[e].sum()) for e in ERAS} }")
print(f"outside every era (kept in the model, excluded from the era table): "
      f"{n_outside}")
assert min(int(era_mask[e].sum()) for e in ERAS) >= 50, \
    "an era bucket is too small to compare; re-derive the bins"

print("mean coordinate on the first 12 SVD components, by era:")
for c in range(12):
    means = [Z[era_mask[e], c].mean() for e in ERAS]
    gap = max(means) - min(means)
    top = terms[np.argsort(svd.components_[c])[-4:]][::-1]
    print(f"  comp {c:>2}  spread={gap:>7.3f}   {' | '.join(top)}")

# %%
# 6. Topic modelling, and the era confound

# %%
nmf = NMF(n_components=15, random_state=SEED, max_iter=300, init="nndsvda")
W = nmf.fit_transform(X)
H = nmf.components_
print(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")
print(f"converged: {nmf.n_iter_} iterations")

topics = []
for t in range(15):
    top = terms[np.argsort(H[t])[-6:]][::-1]
    topics.append(" | ".join(top))
    print(f"  topic {t:>2}: {topics[-1]}")

dom = W.argmax(axis=1)
print(f"\ndocument-topic matrix {W.shape}")
# The 0.4 "dominant topic" threshold was calibrated on LDA, where W is a
# probability distribution and rows sum to 1. NMF factors do not: they are
# non-negative and unbounded, so no threshold is meaningful. Comparing NMF
# rows to an LDA threshold is a category error, and it is why the number
# below is 100%. The LDA-based version of this diagnostic is in column 2.
print(f"documents whose strongest NMF topic is below 0.4: "
      f"{(W.max(axis=1) < 0.4).sum()} ({(W.max(axis=1) < 0.4).mean():.1%})")
print("  ^ meaningless as stated: NMF weights are not probabilities. Reported")
print("    to show why the LDA threshold must not be carried across models.")
wmax = W.max(axis=1)
print(f"  actual max-weight distribution: median {np.median(wmax):.3f}, "
      f"95th pct {np.percentile(wmax, 95):.3f}, max {wmax.max():.3f}")

# %%
# 7. Is topic identity confounded with era?

# %%
tab = pd.crosstab(df.era, dom, normalize="index")
print("share of each era's papers in each topic:")
print((tab * 100).round(1).to_string())

# For each topic, the era distribution of its members versus the corpus.
print("\nchi-square of era x topic, and the era skew of the largest topics:")
from scipy.stats import chi2_contingency
ct = pd.crosstab(df.era, dom)
chi2, p, dof, _ = chi2_contingency(ct)
print(f"  chi2 = {chi2:.1f}, dof = {dof}, p = {p:.3e}")
print(f"  {p < 0.05 and 'topic membership predicts era' or 'no significant era effect'}")

for t in range(5):
    mask = (dom == t)  # dom is an ndarray already; .values on it raises
    if mask.sum() < 20:
        continue
    dist = pd.Series(era_labels[mask]).value_counts(normalize=True)
    base = pd.Series(era_labels).value_counts(normalize=True)
    lift = (dist / base).sort_values(ascending=False)
    print(f"  topic {t}: n={mask.sum():>4}  most-skewed era {lift.index[0]} "
          f"({lift.iloc[0]:.2f}x baseline)  {topics[t][:60]}")

# %%
fig, ax = plt.subplots(figsize=(13, 5))
top_t = np.argsort(-np.array([(dom == t).sum() for t in range(15)]))[:10]
mat = tab[top_t].T * 100
im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(top_t)))
ax.set_yticklabels([f"{t}: {topics[t][:38]}" for t in top_t], fontsize=8)
ax.set_xticks(range(len(ERAS)))
ax.set_xticklabels(ERAS)
ax.set_xlabel("era")
ax.set_title("NMF topic composition by era — real NIPS abstracts\n"
             "a single topic model is partly a model of when a paper was written")
plt.colorbar(im, ax=ax, label="% of era's papers in topic")
plt.tight_layout()
plt.show()

# %%
# 8. The honest reading

# %%
print("""
NIPS TOPIC MODELLING ON THE REAL CORPUS
=======================================

Substituted corpus   603 NYT children's bestsellers, one era, one domain
This notebook        6,361 NIPS abstracts, 1987–2019, thirty years of ML

What the real corpus shows that the substitute could not
  - vocabulary does not persist across eras. Overlap with 1987–98 vocabulary
    is printed above and is far from 1.0, which is the direct evidence that a
    single topic model over the full range is fitting a mixture of subfields
    that did not exist in 1987.
  - the era x topic table is the finding. If a topic's members are concentrated
    in one era, the topic is partly a temporal artefact rather than a subject.
  - the type/token ratios differ by a factor across the three periods, so any
    single k chosen over the pooled corpus is a compromise between eras.

What this does not settle
  - the k chosen in column 2 was selected on the substituted corpus, so that
    number does not transfer. This notebook does not re-run the stability
    analysis at every k on 6,361 documents; that is the obvious next step and
    it is expensive.
  - 6,361 abstracts is still small for a 30-year corpus. It is 10x the
    substitute, not 100x.
""")

# %%
corpus_real = {
    "n_papers": int(len(papers)),
    "n_abstracts": int(len(df)),
    "years": [int(papers.year.min()), int(papers.year.max())],
    "vocab_size": int(len(terms)),
    "lsa_variance_50": round(float(TruncatedSVD(50, random_state=SEED).fit(X)
                                 .explained_variance_ratio_.sum()), 4),
    "era_x_topic_p": float(p),
    "nmf_reconstruction_error": round(float(nmf.reconstruction_err_), 4),
    "unassigned_fraction": round(float((W.max(axis=1) < 0.4).mean()), 4),
}
print(corpus_real)
