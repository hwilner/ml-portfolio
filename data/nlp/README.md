# NLP data

| File | Source | Status |
|---|---|---|
| `babynames_nysiis.csv` | US SSA baby names, NYSIIS-normalised | committed |
| `nytkids_yearly.csv` | NYT children's bestsellers, 2008–2017 | committed |
| `nytkids/` | the bestseller text itself | committed |
| `nips/papers.csv` | Kaggle `rowhitswami/nips-papers-1987-2019-updated` | **fetch, 325 MB** |

## Fetching the NIPS papers

```bash
kaggle datasets download -d rowhitswami/nips-papers-1987-2019-updated \
    -p data/nlp/nips
cd data/nlp/nips && unzip *.zip
```

9,680 papers, 1987–2019. **Abstract coverage starts in 2007** — only 4 papers
before that carry one, though 3,124 have full text. Any analysis over "1987–2019"
is really over 2007–2019 and will undercount the early field; see
`notebooks/07-retrospective/04-nlp-topic-modelling/04-real-data-nips-kaggle.ipynb`.

Third-party data, so it is fetched rather than committed. Verify the licence
before reuse.
