# Natural language processing

Two projects in text processing. Both execute end to end.

---

## 1. The hottest topics in machine learning

**File:** [`02-hottest-topics-lda.ipynb`](02-hottest-topics-lda.ipynb)
**Status:** ✅ runs end to end
**Competencies:** unsupervised learning ●● · text preprocessing ●·

Unsupervised topic discovery over NIPS paper titles using **Latent Dirichlet
Allocation**.

**Approach:** count submissions per year to show the field's growth → clean
titles (strip punctuation, lowercase) → word cloud to sanity-check the
cleaning → `CountVectorizer` with English stop words → LDA with 14 topics →
print the top words per topic.

### What was fixed

| Problem | Fix |
|---|---|
| The notebook loaded `datasets/papers.csv`, but the repository only ever held `papers.csv.gz.part_aa` … `.part_ah` — a 110 MB multi-part download **never reassembled** | Those fragments are deleted. The corpus is now a manual step (200 MB exceeds GitHub's raw-file limit, so there is no stable link); without it the notebook builds a **clearly-labelled synthetic** corpus so the LDA code still runs |
| `count_vectorizer.get_feature_names()` | Removed in scikit-learn 1.0 → `get_feature_names_out()` |
| `from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS` | Raises from scikit-learn 1.2 → `stop_words="english"` |
| `re.sub('[,\.!?]', '', x)` | A character class removing only four marks, and a `TypeError` on a missing title → Unicode-aware regex plus an explicit null guard |
| `wordcloud.to_image()` | Returns a PIL image that Jupyter never displays → plotted properly |
| No vocabulary pruning | Added `min_df=2`, `max_df=0.5` and a letters-only token pattern |

### Getting the real data

```bash
python ../../scripts/fetch_data.py --only lda    # prints the Kaggle steps
```

Until you do, the topics printed describe invented text. The notebook says so on
every figure.

### Why LDA is worth learning properly

LDA treats each document as a distribution over topics, and each topic as a
distribution over words. It is a generative model with an explicit
probabilistic structure — which makes it far more interpretable than, say, NMF
on the same data, and a good bridge from counting to probabilistic modelling.

---

## 2. Name game: gender prediction via NYSIIS

**File:** [`01-name-gender-prediction-nysiis.ipynb`](01-name-gender-prediction-nysiis.ipynb)
**Status:** ✅ runs end to end — **the method is still weak, on purpose-documented grounds**
**Competencies:** string algorithms ●● · data joining ●·

Predicting the gender of New York Times best-selling authors from their first
names, by encoding names phonetically with the **New York State Intelligence
System** algorithm and matching them against Social Security baby-name
statistics.

### What is worth learning

The mechanics, not the conclusion. NYSIIS folds names that sound alike toward a
shared key, and the notebook measures the cost of that folding directly by
comparing unique-name counts before and after. Fuzzy matching across two
datasets on a computed join key is a real technique, and the SSA lookup is a
plausible data source.

### Why the conclusion is not trustworthy

Beyond the incomplete steps, the method is weak on its own terms:

- **The collisions are severe.** The notebook demonstrates that `beach` and
  `bitch` produce the same NYSIIS key. Folding that aggressively throws away
  the distinctions the task needs.
- **The join is a single point of failure.** One bad name match assigns a wrong
  gender with no mechanism to catch it, and there is no confidence threshold on
  the match.
- **The source data is not what it appears.** SSA baby-name data is derived
  from *applications* for a Social Security number, which skews toward
  populations that applied, and gives no per-name gender beyond the aggregate
  split.
- **The stated finding is not evidence.** "More authors are female" follows
  from the join quality, not from anything about publishing.

Gender inference from names is also ethically fraught, and the technique is
known to be unreliable for exactly the groups this kind of analysis tends to
get wrong.

Keep this project for the string algorithms and the fuzzy-join mechanics. Do
not cite it as a modelling result.

### What was fixed

| Problem | Fix |
|---|---|
| Steps 6–8 were unfilled `...YOUR CODE ...` placeholders | Written: per-year counts via `groupby().unstack()`, a stacked bar chart, and a grouped chart with the year axis offset by 0.4 |
| `years = ...` / `years_shifted = ...` | `...` is an `Ellipsis`, not a value — replaced with real arithmetic |
| `import fuzzy` raised `ImportError` for most people | `fuzzy` ships a C extension needing a compiler; now resolved through a shim with a pure-Python fallback |
| `KeyError: 'year'` | The source column is `Year` |

The method itself would need a different join strategy — a probabilistic name
classifier over raw first names rather than a deterministic phonetic lookup.
The notebook states that in its own output rather than only in this file.
