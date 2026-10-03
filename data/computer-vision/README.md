# Computer-vision data

| File | Source | Status |
|---|---|---|
| `naive-bees/labels.csv` | the 2019 BeeImage subset: 500 rows of `id, genus` | committed |
| `naive-bees/images/` | **the 2019 JPEGs — irrecoverably corrupted, deleted** | see `BUGFIXES.md` |
| `bees/bee_data.csv` | Kaggle `jenny18/honey-bee-annotated-images` | committed (497 KB) |
| `bees/bee_imgs/` | the 5,172 photographs | **fetch, 50 MB** |

## Fetching the photographs

```bash
kaggle datasets download -d jenny18/honey-bee-annotated-images \
    -p data/computer-vision/bees --unzip
```

This is the dataset the 2019 notebook was written against, recovered. It
annotates subspecies, health, pollen load and caste — **not genus**, so the
original genus task is not reproducible from it and
`notebooks/07-retrospective/05-computer-vision/04-real-data-kaggle-bees.ipynb`
runs health classification instead and says so.

Third-party images, so they are fetched rather than committed.
