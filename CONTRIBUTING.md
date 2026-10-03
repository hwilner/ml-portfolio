# Contributing

## Two kinds of content

This repository holds two different kinds of work, with different standards.

**`src/ml_from_scratch/`** is a package. It must be correct, tested and
documented. Changes need to keep the bar.

**`notebooks/`** is exploratory. It is allowed to be exploratory, but not to be
*misleading*. Every notebook must execute end to end, and a notebook that needs
data it cannot download must say so on every figure - not only in this file.

## Rules for the package

1. **NumPy only.** No scikit-learn, no TensorFlow. The entire point is that the
   algorithms are visible rather than delegated.
2. **Every public function and class needs a Google-style docstring** with a
   one-line summary, `Args:`, `Returns:`, `Raises:` and a runnable `Example:`.
   See <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>.
3. **Every bug fix needs a test whose name says what broke.** Several existing
   tests are named for the notebook defect they guard against — do not write
   `test_predict_2`. Write `test_feature_zero_is_usable`.
4. **Types on public signatures.** NumPy-style types; the modules use
   `from __future__ import annotations` so `X | None` works on Python 3.9.
5. **No mutation of caller data.** Setters return new arrays. The original
   notebook rescaled in place, which silently corrupted inputs on every second
   call; `test_does_not_mutate_input` exists to stop that coming back.

## Before opening a pull request

```bash
pytest tests/ -v                          # package tests
python -m pytest --doctest-modules src/   # package doctests
python -m doctest scripts/fetch_data.py   # fetch-script doctests
jupyter nbconvert --to notebook --execute --stdout notebooks/**/*.ipynb > /dev/null
```

All of it must be green, **including the notebooks**. Reading a notebook is not
verification - three of the worst bugs in this repository's history
(`BUGFIXES.md`) raised no exception at all. They were only visible in the
printed numbers.

## Adding a notebook

Put it in the right numbered folder with a descriptive kebab-case name:
`03-machine-learning/04-your-project.ipynb`.

Then:

- [ ] Add a first markdown cell stating what it does and whether it runs.
- [ ] If it does not run, add a **Known issues** entry in the root
      `README.md`, with the actual error.
- [ ] If it uses a dataset over ~1 MB, add it to `scripts/fetch_data.py` and
      `.gitignore` rather than committing it.
- [ ] Update the relevant row in [`SKILLS.md`](SKILLS.md).
- [ ] Strip outputs? No — keep them. They are evidence.

## Adding to the skills matrix

[`SKILLS.md`](SKILLS.md) is meant to be checkable, so:

- Be honest about level. If a technique ran once and nothing was verified, it
  is ●, not ●●.
- A method that does not run does not count as demonstrated, however good the
  thinking behind it.
- If you find a gap, add it to the [Gap analysis](SKILLS.md#gap-analysis). A
  documented gap is more useful to a reader than a quietly omitted one.

## The retrospective notebooks

`notebooks/07-retrospective/` holds 18 notebooks: six topics, three columns
each (2019 as written / 2019 with judgement / 2026 tools).

**Each notebook is generated from a `.py` source file** alongside it. Edit the
`.py`, never the `.ipynb` — the notebook is a build artefact and your changes
will be overwritten.

```bash
python scripts/build_notebook.py                    # compile all .py -> .ipynb
python scripts/build_notebook.py 01-causal-inference # one topic
python scripts/run_notebooks.py 07-retrospective    # execute, fail on error
```

`run_notebooks.py` is the gate. A notebook that does not execute is not part of
this repository, and a claim in a markdown cell that no code cell supports is
not a claim.

Two conventions the notebooks follow:

- **Synthetic data is labelled synthetic**, in the variable names, the figure
  titles and the prose. Where a real dataset was unavailable, the notebook says
  so and states what the result does and does not demonstrate.
- **A number is printed next to the claim it supports.** If a cell asserts
  something, a cell nearby computed it.

## Datasets

Two third-party corpora are fetched rather than committed, because they are
other people's data and together they are 375 MB:

```bash
# real NIPS abstracts  -> data/nlp/nips/
kaggle datasets download -d rowhitswami/nips-papers-1987-2019-updated \
    -p data/nlp/nips && cd data/nlp/nips && unzip *.zip

# real bee photographs -> data/computer-vision/bees/
kaggle datasets download -d jenny18/honey-bee-annotated-images \
    -p data/computer-vision/bees --unzip
```

Both notebooks raise a clear `SystemExit` with these instructions if the data is
absent, rather than failing obscurely or silently falling back to synthetic data.
The metadata CSVs *are* committed, so the notebooks can at least report what the
corpus contains before you fetch the bulk.

Everything else — including all synthetic corpora — is generated in-notebook and
is fully reproducible.
