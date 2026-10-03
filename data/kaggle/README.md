# Kaggle competition data

Not committed — 2019 competitions, and the data belongs to Kaggle. Fetch with:

```bash
export KAGGLE_CONFIG_DIR=~/.kaggle        # needs kaggle.json or KAGGLE_API_TOKEN
for c in forest-cover-type-kernels-only dont-overfit-ii homesite-quote-conversion; do
  kaggle competitions download -c "$c" -p "data/kaggle/$c"
  (cd "data/kaggle/$c" && for z in *.zip; do unzip -o "$z"; done)
done
```

| Competition | Size | Used by |
|---|---|---|
| `forest-cover-type-kernels-only` | 15 MB | `01-forest-cover-type` |
| `dont-overfit-ii` | 37 MB | `02-dont-overfit` |
| `homesite-quote-conversion` | 62 MB | `03-homesite-quote` |
| `optiver-trading-at-the-close` | 640 MB | not fetched; 3 GB RAM is the constraint, not the download |

The notebooks locate data through `scripts/kaggle_data.py`, which searches the
Kaggle-native `/kaggle/input/...` path, this directory, and `$KAGGLE_DATA_DIR`.
Every notebook raises with the fetch command if the data is absent.
