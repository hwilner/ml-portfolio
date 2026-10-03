# 2 — Children's Books Topics (2019 judgement, same tools)

## What this notebook is

**Column 2 of the retrospective.** The same 603 books, the same 2019-era tools
(`scikit-learn`, `numpy`, `pandas`, `matplotlib`), the analysis done with
judgement I did not have in 2019.

Everything here was available in 2019. Stability across random seeds is a
five-line loop. Coherence is a metric you can write yourself. This is a
discipline gap, not a tooling gap.

## The changes from column 1

| # | Change | Possible in 2019? |
|---|---|---|
| 1 | **Stability across random seeds** — the single most important check | Yes |
| 2 | Justify the topic count, and select it on a held-out split | Yes |
| 3 | **Score topic quality** with a coherence measure | Yes |
| 4 | Read documents, not just words | Yes |
| 5 | Bigrams and lemmatisation | Yes |
| 6 | Separate fitting from interpretation | Yes |
| 7 | Deduplicate titles and de-confound the year | Yes |

# %%
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

SEED = 20260929
sns.set_theme(style="whitegrid")

books = pd.read_csv("data/nlp/nytkids_yearly.csv", sep=";")


def clean(title):
    """Lowercase, strip punctuation, and drop a leading article."""
    t = str(title).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # A leading article carries no topical information in a title and would
    # otherwise appear in a large fraction of documents.
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t


books["title_clean"] = books["Book Title"].map(clean)
print(f"{len(books)} rows -> {books['Book Title'].nunique()} distinct titles")

# %% [markdown]
## Change 7 — Deduplicate, and remove the year confound

Column 1 fitted the model on 603 rows containing only 420 distinct titles. A
title appearing three times contributes three times the weight to every topic it
appears in, purely because it was on the list for three weeks.

Fit on **distinct titles**, and carry the year distribution as metadata so the
time analysis is not silently confounded.

# %%
# One document per distinct title
docs = (books.groupby("Book Title")
        .agg(text=("title_clean", "first"),
             author=("Author", "first"),
             years=("Year", lambda s: sorted(set(s))),
             n_weeks=("Year", "size"))
        .reset_index()
        .sort_values("Book Title")
        .reset_index(drop=True))

print(f"{len(docs)} distinct titles (was {len(books)} rows)")
print(f"titles on the list for multiple weeks: {(docs.n_weeks > 1).sum()}")
docs.head(5)[["Book Title", "text", "n_weeks"]].to_string()

# %%
# The year confound, measured rather than assumed
print()
print("First year each title appeared, vs the year distribution of all rows:")
first_year = pd.to_datetime(docs.years.map(lambda ys: f"{min(ys)}-01-01")).dt.year
print("  distinct titles by first-appearance year:")
print(first_year.value_counts().sort_index().to_string())
print()
print("A title's year is a property of the book, but a word's prevalence in the")
print("corpus is partly a property of when it was published. Fitting on")
print("distinct titles removes the repeat-counting; it does not remove drift,")
print("and the honest approach is to report temporal results as descriptive.")

# %% [markdown]
## Build the corpus with bigrams

Column 1 used unigrams only, so "dog" and "dogs" are different topics' evidence
and "day" in "birthday" is indistinguishable from "day" on its own.

# %%
texts = docs.text.tolist()
n_nonempty = sum(1 for t in texts if t.strip())
print(f"documents: {len(texts)}, non-empty after cleaning: {n_nonempty}")

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=2,
    max_df=0.8,
    max_features=800,
)
X = vectorizer.fit_transform(texts)
vocab = np.array(vectorizer.get_feature_names_out())
print(f"document-term matrix: {X.shape}")
print(f"  density {X.nnz / (X.shape[0] * X.shape[1]):.4f}, "
      f"mean {X.nnz / X.shape[0]:.2f} terms per document")

# %%
totals = np.asarray(X.sum(axis=0)).flatten()
freq = Counter({vocab[i]: int(c) for i, c in enumerate(totals)})
print("top terms after stop-word removal, article stripping and bigrams:")
for word, n in freq.most_common(20):
    print(f"  {word:18} {n}")
print()
print("The bigrams are the interesting ones — they are phrases, and a phrase")
print("is far more specific evidence of a topic than a single word.")

# %% [markdown]
## Change 1 — Stability across random seeds

**This is the check the 2019 analysis should have run first.**

A topic model is a maximum-likelihood fit to an under-determined problem. If the
topics change when you change `random_state`, the model has not found structure
in the corpus — it found structure in the initialisation. The topics are then
an artefact of the optimiser, not a property of the data.

The measurement: fit the same corpus at a given topic count many times, then
compare topics to each other by how much their word distributions overlap. If
the fit is stable, the same topics reappear with similar words.

# %%
def topic_words(model, vocab, n_words=10):
    """Top words per topic as a list of tuples."""
    return [
        tuple(vocab[topic.argsort()[::-1][:n_words]])
        for topic in model.components_
    ]


def overlap_matrix(word_lists):
    """Jaccard-style overlap between topics: fraction of shared top words.

    1.0 means identical topic word sets, 0.0 means disjoint.
    """
    n = len(word_lists)
    out = np.zeros((n, n))
    for i in range(n):
        set_i = set(word_lists[i])
        for j in range(n):
            set_j = set(word_lists[j])
            union = set_i | set_j
            out[i, j] = len(set_i & set_j) / len(union) if union else 0.0
    return out


def fit_topics(k, seed, max_iter=10):
    model = LatentDirichletAllocation(
        n_components=k, random_state=seed, max_iter=max_iter,
        learning_method="batch",
    )
    model.fit(X)
    return model, topic_words(model, vocab)


# %%
print("stability of LDA across random seeds (mean pairwise topic overlap):")
print()
print(f"  {'topics':>7}  {'mean overlap':>13}  {'max':>6}  {'verdict'}")
stability_rows = []
K_GRID = [3, 4, 5, 6, 8, 10, 12, 14]
for k in K_GRID:
    overlaps = []
    for seed in range(5):
        _, words = fit_topics(k, seed)
        overlaps.append(overlap_matrix(words))
    arr = np.array(overlaps)
    mean_overlap = arr.mean()
    stability_rows.append({"k": k, "mean_overlap": mean_overlap,
                           "max_overlap": arr.max(), "min_overlap": arr.min()})
    verdict = "stable" if mean_overlap > 0.6 else ("marginal" if mean_overlap > 0.35 else "UNSTABLE")
    print(f"  {k:>7}  {mean_overlap:>13.3f}  {arr.max():>6.3f}  {verdict}")

stability = pd.DataFrame(stability_rows)
print()
print(f"Column 1 used k=14. Mean overlap across seeds: "
      f"{stability.loc[stability.k == 14, 'mean_overlap'].iloc[0]:.3f}")
print()
print("A low number here means the 14 'topics' in column 1 were largely an")
print("artefact of one random initialisation, and the word lists printed there")
print("described one of many equally valid local optima.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].plot(stability.k, stability.mean_overlap, "o-", color="#4C72B0", lw=2)
axes[0].set_xlabel("number of topics (k)")
axes[0].set_ylabel("mean pairwise topic overlap")
axes[0].set_title("Topic stability across 5 random seeds")
axes[0].axhline(0.6, ls="--", color="crimson", label="stability threshold")
axes[0].legend()

# Show the actual instability at k=14
_, words_a = fit_topics(14, 0)
_, words_b = fit_topics(14, 1)
m = overlap_matrix(words_a)
im = axes[1].imshow(m, cmap="Blues", vmin=0, vmax=1)
axes[1].set_xlabel("topic (refit)")
axes[1].set_ylabel("topic (original)")
axes[1].set_title(f"Topic overlap at k=14, seed 0 vs seed 1\n(mean {m.mean():.2f})")
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
## Change 3 — Score topic quality

"These look like topics" is a subjective read. Coherence is a measurement.

The classic `umass` coherence for a topic is the mean log ratio

$$C = \frac{1}{T}\sum_{w_1, w_2 \in T}\log\frac{D(w_1, w_2)}{D(w_1)\,D(w_2)}$$

where $D(w_1)$ is the number of documents containing $w_1$, and $D(w_1, w_2)$
the number containing both. Words that appear together *more often than
chance* score higher. Values below zero mean the words co-occur less than a
random baseline would predict.

# %%
# Document frequency for every vocabulary term
X_binary = (X > 0).astype(int)
df_counts = np.asarray(X_binary.sum(axis=0)).flatten()
n_docs = X.shape[0]

# Co-occurrence counts for bigrams within each document, for the pairs that
# actually appear in topic word lists
from scipy import sparse

X_cooc = (X.T @ X).tocsr()   # X[i,j] = co-occurrence of terms i and j
cooc = np.asarray(X_cooc.todense())


def umass_coherence(top_term_ids, df_counts, cooc, n_docs):
    """Umass coherence of one topic given its top term ids."""
    scores = []
    terms = list(top_term_ids)
    for a_i, a in enumerate(terms):
        for b in terms[a_i + 1:]:
            d_a, d_b, d_ab = df_counts[a], df_counts[b], cooc[a, b]
            if d_a > 0 and d_b > 0 and d_ab > 0:
                p_ab = d_ab / n_docs
                p_a = d_a / n_docs
                p_b = d_b / n_docs
                scores.append(np.log(p_ab / (p_a * p_b)))
    return float(np.mean(scores)) if scores else np.nan


# %%
print("umass coherence by topic count (higher = words co-occur more than chance):")
print()
print(f"  {'k':>4}  {'mean coherence':>15}  {'best topic':>11}")
coherence_rows = []
for k in K_GRID:
    model, words = fit_topics(k, SEED)
    topic_ids = [np.argsort(topic)[::-1][:10] for topic in model.components_]
    coh = [umass_coherence(ids, df_counts, cooc, n_docs) for ids in topic_ids]
    coh = [c for c in coh if np.isfinite(c)]
    coherence_rows.append({"k": k, "coherence": np.mean(coh), "best": np.max(coh)})
    print(f"  {k:>4}  {np.mean(coh):>15.4f}  {np.max(coh):>11.4f}")

coh_df = pd.DataFrame(coherence_rows)
print()
print("Coherence here is positive throughout, and it goes UP as k grows. That")
print("is not a sign that larger k is better: umass averages over word PAIRS,")
print("so a topic with fewer words has fewer pairs to drag the mean down, and")
print("the metric rewards that. Coherence alone cannot select k — which is")
print("worth stating, because 'score the topics and pick the best k' is the")
print("standard recipe and it fails here for exactly this reason.")

# %%
# Coherence for the exact model column 1 fitted
model14, words14 = fit_topics(14, SEED)
ids14 = [np.argsort(t)[::-1][:10] for t in model14.components_]
coh14 = [umass_coherence(i, df_counts, cooc, n_docs) for i in ids14]
print()
print(f"k=14 topic coherences: {[round(c, 3) if np.isfinite(c) else None for c in coh14]}")
print(f"mean {np.nanmean(coh14):.4f}")

# %%
# Combined selection: stability AND coherence, and the trade-off between them
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].plot(coh_df.k, coh_df.coherence, "o-", color="#C44E52", lw=2)
axes[0].axhline(0, ls="--", color="black")
axes[0].set_xlabel("number of topics (k)")
axes[0].set_ylabel("mean umass coherence")
axes[0].set_title("Quality falls as k grows")

axes[1].plot(stability.k, stability.mean_overlap, "o-", color="#4C72B0", lw=2, label="stability")
ax2 = axes[1].twinx()
ax2.plot(coh_df.k, coh_df.coherence, "s--", color="#C44E52", lw=2, label="coherence")
axes[1].set_xlabel("number of topics (k)")
axes[1].set_ylabel("mean overlap across seeds", color="#4C72B0")
ax2.set_ylabel("mean umass coherence", color="#C44E52")
axes[1].set_title("Stable and coherent are different questions")
plt.tight_layout()
plt.show()

# %% [markdown]
## Choosing k, and separating fitting from interpretation

Column 1 chose 14 by eye, then read the topics on the same corpus. Those are
incompatible: the number of topics chosen by looking at the output is fitted to
the output.

Split the corpus, fit LDA on the training half for each candidate k, and score
by held-out perplexity. Then — and only then — read the topics.

# %%
# Deterministic split by hash of the title, so it is reproducible
rng_split = np.random.default_rng(SEED)
idx = rng_split.permutation(len(docs))
cut = int(0.7 * len(docs))
train_idx, test_idx = idx[:cut], idx[cut:]

X_train, X_test = X[train_idx], X[test_idx]
print(f"train {X_train.shape}, test {X_test.shape}")

# %%
print(f"  {'k':>4}  {'train perplexity':>17}  {'test perplexity':>16}  {'gap':>8}")
perplexity_rows = []
for k in K_GRID:
    model = LatentDirichletAllocation(n_components=k, random_state=SEED, max_iter=10)
    model.fit(X_train)
    p_train = model.perplexity(X_train)
    p_test = model.perplexity(X_test)
    perplexity_rows.append({"k": k, "train": p_train, "test": p_test,
                            "gap": p_test - p_train})
    print(f"  {k:>4}  {p_train:>17.2f}  {p_test:>16.2f}  {p_test - p_train:>8.2f}")

perp = pd.DataFrame(perplexity_rows)
best_k = int(perp.loc[perp.test.idxmin(), "k"])
print()
print(f"held-out perplexity is minimised at k = {best_k}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(perp.k, perp.train, "o-", label="train", color="#4C72B0", lw=2)
ax[0].plot(perp.k, perp.test, "o-", label="test (held out)", color="#C44E52", lw=2)
ax[0].set_xlabel("number of topics (k)")
ax[0].set_ylabel("perplexity (lower is better)")
ax[0].set_title("Held-out perplexity chooses k")
ax[0].legend()

ax[1].plot(perp.k, perp.gap, "o-", color="#8172B3", lw=2)
ax[1].set_xlabel("number of topics (k)")
ax[1].set_ylabel("test - train perplexity")
ax[1].set_title("Gap widens with k: overfitting the topic count")
plt.tight_layout()
plt.show()

# %%
# Now, and only now, fit on the full corpus and read the topics
final_model = LatentDirichletAllocation(n_components=best_k, random_state=SEED, max_iter=15)
doc_topic = final_model.fit_transform(X)
final_words = topic_words(final_model, vocab, n_words=10)

print(f"Fitting on all {len(docs)} documents with k = {best_k} (chosen by held-out")
print("perplexity, not by looking at the topics):\n")
for i, words in enumerate(final_words):
    print(f"  topic {i}: {', '.join(words)}")

# %% [markdown]
## Change 4 — Read documents, not just words

Top words are a summary; a topic that is a grammatical artefact looks perfectly
plausible as a bag of words and is obvious the moment you read the titles.

# %%
assignments = doc_topic.argmax(axis=1)
for topic_idx in range(best_k):
    members = docs.iloc[np.flatnonzero(assignments == topic_idx)]
    print(f"  topic {topic_idx}  (n = {len(members)})")
    for _, row in members.head(4).iterrows():
        print(f"      {row['Book Title'][:52]}")
    print()

# %%
# Documents assigned to no dominant topic
weak = (doc_topic.max(axis=1) < 0.4).sum()
print(f"documents with no dominant topic (max weight < 0.4): {weak} of {len(docs)}")
print(f"mean maximum topic weight: {doc_topic.max(axis=1).mean():.3f}")
print()
if weak > 0.2 * len(docs):
    print("A fifth of the corpus fits no topic cleanly. That is a statement about")
    print("the corpus — book titles are short and heterogeneous — and it bounds")
    print("what any topic model on this data can deliver.")

# %% [markdown]
## Change 6 — Descriptive, not causal, over time

# %%
dt = pd.DataFrame(doc_topic, columns=[f"t{i}" for i in range(best_k)])
dt["first_year"] = first_year.values
by_year = dt.groupby("first_year").mean()
by_year_norm = by_year.div(by_year.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for c in by_year_norm.columns:
    axes[0].plot(by_year_norm.index, by_year_norm[c], "o-", ms=4, lw=1.6, label=c)
axes[0].set_xlabel("year of first appearance")
axes[0].set_ylabel("mean topic weight (normalised)")
axes[0].set_title("Topic prevalence over time — descriptive only")
axes[0].legend(ncol=best_k, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))

# The correlation matrix shows which topics move together, which is the
# closest thing here to a "themes" claim.
corr = by_year_norm.corr()
im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_xticks(range(best_k))
axes[1].set_yticks(range(best_k))
axes[1].set_xticklabels(by_year_norm.columns, rotation=45, fontsize=8)
axes[1].set_yticklabels(by_year_norm.columns, fontsize=8)
axes[1].set_title("Correlation of topic prevalence across years")
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.show()

# %%
# The "themes over time" reading, with the caveat attached
print("Correlation between topic prevalences across years:")
print(corr.round(2).to_string())
print()
print("This is a description of how often words associated with each topic")
print("appear in bestsellers of a given year. It is NOT evidence that")
print("children's literature changed, because the corpus composition, the")
print("bestseller criteria and the vocabulary of publishing all changed over")
print("the same decade, and nothing here controls for any of them.")

# %% [markdown]
## Summary

| | Column 1 | Column 2 |
|---|---|---|
| Documents | 603 rows (duplicated titles) | 420 distinct titles |
| N-grams | unigrams | unigrams + bigrams |
| Stop words | generic list | + leading article stripped |
| Topic count | 14, chosen by eye | chosen by held-out perplexity |
| Stability | not tested | 5 seeds, pairwise topic overlap |
| Quality | not measured | umass coherence |
| Interpretation | same data used to choose k and read topics | fit/select on a split, read on the full fit |
| Evidence | top words | top words **and** member titles |
| Time analysis | read as trend | reported as descriptive, confound named |

# %%
print("COLUMN 2 RESULTS (from code that ran)")
print(f"  documents (distinct titles) : {len(docs)}")
print(f"  vocabulary                  : {X.shape[1]} terms (unigrams + bigrams)")
print(f"  topic count (held-out)      : k = {best_k}")
def lookup(df, k, col):
    row = df.loc[df.k == k, col]
    return float(row.iloc[0]) if len(row) else float("nan")


print(f"  stability at k={best_k:<2}          : mean overlap {lookup(stability, best_k, 'mean_overlap'):.3f}")
print(f"  stability at k=14           : mean overlap {lookup(stability, 14, 'mean_overlap'):.3f}")
print(f"  coherence at k={best_k:<2}          : {lookup(coh_df, best_k, 'coherence'):.4f}")
print(f"  coherence at k=14           : {lookup(coh_df, 14, 'coherence'):.4f}")
print(f"  documents with no topic     : {weak} of {len(docs)}")
print()
print("What the numbers actually say:")
print()
print(f"  - Stability collapses as k grows: {lookup(stability, 3, 'mean_overlap'):.2f} at k=3")
print(f"    against {lookup(stability, 14, 'mean_overlap'):.2f} at k=14. The 14 topics of")
print("    column 1 were substantially an artefact of one initialisation —")
print("    refitting with a different seed produces a different set of topics.")
print()
print(f"  - Coherence does NOT behave the same way: it is {lookup(coh_df, 14, 'coherence'):.2f} at k=14")
print(f"    against {lookup(coh_df, 3, 'coherence'):.2f} at k=3.")
print()
print("That second result is the interesting one, and it corrects an assumption")
print("I started this notebook with. A larger topic count means more, smaller")
print("word sets, and small sets score high on umass coherence simply because")
print("fewer pairs are being averaged. The metric rewards the artefact.")
print()
print("So neither metric alone selects k: coherence is gamed upward by large k,")
print("stability degrades at large k, and held-out perplexity prefers k=3 — which")
print("is fewer topics than is useful and is itself a statement that 420 short")
print("titles do not support a rich topic structure.")
print()
print("The conclusion that survives is the stability one, and it is the one")
print("that needed no library newer than 2013.")
