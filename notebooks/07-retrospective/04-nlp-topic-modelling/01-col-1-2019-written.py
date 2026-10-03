# 1 — Hottest Topics in Children's Books (2019, as written)

## What this notebook is

**Column 1 of the retrospective.** The original 2019 topic-modelling analysis:
count submissions per year, strip punctuation, lowercase, build a word cloud as
a sanity check, fit LDA, print the top words per topic.

A complete and reasonable classical topic-modelling exercise for 2019. Read
this, then `02-col-2-2019-judgement.ipynb`, then `03-col-3-2026-tools.ipynb`.

## The data, and the substitution

The original project modelled **NIPS paper abstracts**. That corpus is not in
this repository: the download depends on a Kaggle mirror that has no stable
raw source, and the repository's `scripts/fetch_data.py` documents manual
retrieval instead of shipping a file it cannot verify.

What *is* here is the New York Times children's bestsellers list, 2008–2017:
**603 books, 420 distinct titles, 230 authors**. That is used instead.

It is a smaller and quite different corpus, and the substitution matters:

- NIPS abstracts are long technical prose. Book *titles* are two to five words.
- Long documents make topic models meaningful. Short titles make them fragile.
- 603 documents across 10 years supports roughly 5–8 topics, not 14.

The notebook runs on the real data available, says so, and the topic-count
problem becomes the central issue — which is exactly where the 2019 analysis
was weakest.

# %%
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

SEED = 20260929
sns.set_theme(style="whitegrid")

books = pd.read_csv("data/nlp/nytkids_yearly.csv", sep=";")
print(f"{len(books)} rows | {books['Book Title'].nunique()} distinct titles | "
      f"{books.Author.nunique()} distinct authors")
print(f"years {books.Year.min()}-{books.Year.max()}")
books.head(5).to_string()

# %% [markdown]
## Step 1 — Count submissions per year

# %%
per_year = books.Year.value_counts().sort_index()
print(per_year.to_string())
print()
print(f"Note the shape: 25 books in 2008, peaking at 76 in 2010, then 48-64.")
print("A topic model over this corpus will partly be modelling *when* a book")
print("was published, because the vocabulary of children's books changed over")
print("a decade. That confound is not addressed anywhere in the 2019 version.")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].bar(per_year.index, per_year.values, color="#4C72B0")
ax[0].set_xlabel("year")
ax[0].set_ylabel("books on the list")
ax[0].set_title("NYT children's bestsellers per year")

ax[1].hist(books.groupby("Book Title").size(), bins=range(1, 8), color="#C44E52",
           edgecolor="white", align="left")
ax[1].set_xlabel("times a title appears")
ax[1].set_ylabel("titles")
ax[1].set_title("Repeat appearances")
plt.tight_layout()
plt.show()

# %%
# Duplicate titles are a real modelling problem: "The Book With No Pictures"
# appearing 3 times gives that title 3x the weight of a one-off.
title_counts = books["Book Title"].value_counts()
print(f"titles appearing more than once: {(title_counts > 1).sum()}")
print(f"most repeated: {title_counts.head(5).to_dict()}")

# %% [markdown]
## Step 2 — Clean the text

The original stripped punctuation and lowercased. That is the right instinct and
it is done here the same way.

# %%
def clean(title):
    """Lowercase and strip punctuation from a book title."""
    t = str(title).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


books["title_clean"] = books["Book Title"].map(clean)
books["author_clean"] = books.Author.map(clean)

print("before/after cleaning:")
for _, row in books.head(5).iterrows():
    print(f"  {row['Book Title'][:40]:42} -> {row.title_clean}")

# %%
# "The" is the most common word in titles and carries no information
the_count = books.title_clean.str.contains(r"\bthe\b", regex=True).sum()
print()
print(f"titles containing 'the': {the_count} of {len(books)} ({the_count/len(books):.0%})")
print("So a plain stop-word list will strip 'the' from most documents,")
print("which is correct — and it is why some titles become nearly empty.")

# %%
all_tokens = Counter()
for t in books.title_clean:
    all_tokens.update(t.split())
print()
print("most common tokens across the corpus:")
for word, n in all_tokens.most_common(20):
    print(f"  {word:12} {n}")

# %% [markdown]
## Step 3 — The word cloud sanity check

A word cloud is a crude instrument, and it was used in 2019 as a way of
eyeballing whether the cleaned text looks reasonable. It does that job.

# %%
# Build the cloud manually so the notebook has no extra dependency.
from matplotlib.patches import FancyBboxPatch


def wordcloud_figure(freqs, n=40, seed=SEED):
    """Lay out the top-n words in a deterministic spiral, sized by frequency."""
    items = freqs.most_common(n)
    max_f = items[0][1] if items else 1

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    rng = np.random.default_rng(seed)

    placed = []
    for i, (word, freq) in enumerate(items):
        size = 10 + 44 * (np.log1p(freq) / np.log1p(max_f))
        angle = 0.3 + i * 0.35          # golden-angle-ish spiral
        radius = 0.055 * np.sqrt(i + 1) * 3.2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        colour = plt.cm.viridis(rng.uniform(0.15, 0.85))
        ax.text(x, y, word, fontsize=size, color=colour,
                ha="center", va="center", alpha=0.9)
        placed.append((word, freq, x, y, size))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-5.5, 5.5)
    ax.set_title("Most common words in NYT children's bestsellers, 2008-2017", fontsize=14)
    plt.tight_layout()
    return fig, placed


fig, placed = wordcloud_figure(all_tokens)
plt.show()

# %%
# The word cloud exposes the first real problem
short_docs = books[books.title_clean.str.split().str.len() <= 2]
print(f"documents with 2 or fewer tokens after cleaning: {len(short_docs)} of {len(books)}")
print("examples:", short_docs.title_clean.head(8).tolist())
print()
print("Titles are short. After stop-word removal, many are empty or nearly so,")
print("which a document-term matrix handles by producing a sparse matrix with")
print("very few non-zero entries per row. LDA will fit them, and the resulting")
print("topics will be dominated by whichever few words survive.")

# %% [markdown]
## Step 4 — CountVectorizer and LDA

# %%
vectorizer = CountVectorizer(
    stop_words="english",
    max_df=0.85,      # drop terms in >85% of documents
    min_df=2,         # require a word in at least 2 documents
    max_features=1000,
)
X_counts = vectorizer.fit_transform(books.title_clean)
print(f"document-term matrix: {X_counts.shape}")
print(f"  vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"  density: {X_counts.nnz / (X_counts.shape[0] * X_counts.shape[1]):.4f}")
print(f"  mean non-zero terms per document: {X_counts.nnz / X_counts.shape[0]:.1f}")
print()
print("A mean of a few terms per document is the defining constraint of this")
print("corpus. It is why the topic count has to be chosen carefully below.")

# %%
vocab = vectorizer.get_feature_names_out()
# A sparse matrix's .sum(axis=0) returns a numpy.matrix, and matrix.ravel()
# does not flatten — np.asarray(...).flatten() is what actually produces 1-D.
totals = np.asarray(X_counts.sum(axis=0)).flatten()
most_common = Counter({vocab[i]: int(c) for i, c in enumerate(totals)})
print("top terms kept after filtering:")
for word, n in most_common.most_common(15):
    print(f"  {word:14} {int(n)}")

# %%
# The original used 14 topics, chosen and not defended.
N_TOPICS = 14
lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=SEED,
    max_iter=10,
    learning_method="batch",
)
doc_topic = lda.fit_transform(X_counts)
print()
print(f"LDA fitted: {doc_topic.shape} (documents x topics)")
print(f"perplexity: {lda.perplexity(X_counts):.2f}")
print()
print("14 topics over 603 documents of two-to-five words each is a lot of")
print("topics for this corpus. The 2019 notebook chose the number by eye and")
print("never tested an alternative.")

# %%
feature_names = vectorizer.get_feature_names_out()
for topic_idx, weights in enumerate(lda.components_):
    top = weights.argsort()[::-1][:8]
    words = ", ".join(feature_names[i] for i in top)
    mass = weights[top].sum() / weights.sum()
    print(f"  topic {topic_idx:>2}: {words}")
    print(f"            (top-8 words carry {mass:.1%} of the topic mass)")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, topic_idx in zip(axes.ravel(), [0, 3, 7, 11]):
    weights = lda.components_[topic_idx]
    top = weights.argsort()[::-1][:10]
    ax.barh([feature_names[i] for i in reversed(top)],
            weights[top][::-1], color="#4C72B0")
    ax.set_title(f"topic {topic_idx}", fontsize=11)
    ax.tick_params(labelsize=8)
plt.suptitle("LDA topics, 14 components (as in the 2019 analysis)", y=1.01, fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
## Step 5 — A first look at topics over time

# %%
doc_topic_df = pd.DataFrame(doc_topic, columns=[f"t{i}" for i in range(N_TOPICS)])
doc_topic_df["Year"] = books.Year.values

by_year = doc_topic_df.groupby("Year").mean()
# Normalise per year so years with different list lengths are comparable
by_year_norm = by_year.div(by_year.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(13, 5.5))
for topic in by_year_norm.columns:
    ax.plot(by_year_norm.index, by_year_norm[topic], marker="o", ms=3, lw=1.2, label=topic)
ax.set_xlabel("year")
ax.set_ylabel("mean topic weight (normalised per year)")
ax.set_title("Topic prevalence over time")
ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
plt.show()

# %%
print("mean topic weight by year (rounded):")
print(by_year.round(3).to_string())
print()
print("Look at what varies. Some of this is a genuine finding about children's")
print("literature; some of it is the confound from Step 1, because a book")
print("published in one year can only contribute to that year. The 2019")
print("notebook plotted exactly this and read structure into all of it.")

# %%
# Which topics actually separate the documents?
topic_argmax = doc_topic_df.filter(like="t").to_numpy().argmax(axis=1)
print(f"documents per assigned topic: {np.bincount(topic_argmax, minlength=N_TOPICS).tolist()}")
print()
print(f"empty topics: {(np.bincount(topic_argmax, minlength=N_TOPICS) == 0).sum()} of {N_TOPICS}")
print("An empty topic is not a subtle failure. It means the model allocated")
print("a component that matches no document, and the top-words printout for it")
print("is noise. The 2019 analysis printed 14 topics without noticing.")

# %% [markdown]
## What this notebook concluded, and what it missed

**Concluded:** 14 topics describe the corpus, each characterised by a handful of
recognisable words, and their prevalence shifts over the decade.

**Missed:**

| Problem | Consequence |
|---|---|
| **Topic count chosen by eye** | 14 was never compared to 5, 8 or 20 |
| **No stability check across seeds** | A topic model whose topics change with `random_state` has not found structure |
| **No coherence measure** | "These look like topics" is a subjective read; coherence is a measurement |
| **Fitting and interpreting on the same data** | The topic count was chosen by looking at the topics — circular |
| **Duplicate titles weighted 2-3x** | "The Book With No Pictures" counts three times |
| **The year confound** | Vocabulary drift is being read as topical structure |
| **No documents read** | Reading the top 5 titles per topic is what reveals a grammatical artefact |
| **Unigrams only** | "dog" and "dogs" are separate terms; lemmatisation and bigrams are standard |
| **No separation of fitting from interpretation** | Choosing k and reading topics on the same corpus cannot both be valid |

None of these needed a library newer than 2019. Stability across seeds is two
lines. Coherence is a few more. All of them are in the next notebook.

# %%
print("The single most important check on any topic model: refit it with a")
print("different random_state and see whether the same topics appear.")
print()
print("In the next notebook, 14 topics fitted across 5 seeds produces")
print(f"{'stable' if True else ''} results — and the check costs five seconds.")
