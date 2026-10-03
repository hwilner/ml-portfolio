"""Locate Kaggle competition data regardless of where it was fetched.

The original notebooks all hardcoded `../input/train.csv`, which only resolves
inside a Kaggle kernel. That path is meaningless on a laptop or in CI, so every
notebook here goes through `competition_path` instead and fails with the fetch
command rather than a bare FileNotFoundError.

Search order:
  1. $KAGGLE_DATA_DIR/<slug>
  2. <repo>/data/kaggle/<slug>
  3. /kaggle/input/<slug>          (inside a Kaggle kernel)
  4. ../input                       (the 2019 relative path, last)
"""
from __future__ import annotations

import os
from pathlib import Path

# This module is BOTH imported as `scripts.kaggle_data` and inlined verbatim
# into built notebooks by scripts/build_notebook.py (a Kaggle kernel receives
# only the notebook file, so an import would fail there). A notebook cell has
# no `__file__`, so resolving the repo root has to tolerate its absence —
# NameError on an undefined global kills the cell with no traceback in the
# run log, which is exactly the kind of failure that costs an hour.
REPO_ROOT = (Path(__file__).resolve().parent.parent  # type: ignore[name-defined]
             if "__file__" in globals() else Path.cwd())


class CompetitionDataMissing(FileNotFoundError):
    pass


def _candidates(slug: str) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("KAGGLE_DATA_DIR")
    if env:
        out.append(Path(env) / slug)
    out.append(REPO_ROOT / "data" / "kaggle" / slug)
    # Verified by probing a real kernel: Kaggle nests COMPETITION data one
    # level deeper than dataset data. A kernel with
    # competition_sources=["forest-cover-type-kernels-only"] mounts at
    #   /kaggle/input/competitions/forest-cover-type-kernels-only/
    # while a kernel with dataset_sources mounts at
    #   /kaggle/input/<slug>/
    # Assuming the flatter path for both is why the first Kaggle run of this
    # notebook raised CompetitionDataMissing on a kernel that had the data
    # attached correctly.
    out.append(Path("/kaggle/input") / "competitions" / slug)
    out.append(Path("/kaggle/input") / slug)
    out.append(REPO_ROOT / "input")
    out.append(REPO_ROOT.parent / "input")
    return out


def competition_dir(slug: str) -> Path:
    """Return the directory holding `slug`'s files, or explain how to get it."""
    for cand in _candidates(slug):
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    tried = "\n".join(f"    {c}" for c in _candidates(slug))
    raise CompetitionDataMissing(
        f"\n\nKaggle competition data not found: {slug!r}\n\n"
        f"Looked in:\n{tried}\n\n"
        f"Fetch it with:\n"
        f"    kaggle competitions download -c {slug} -p data/kaggle/{slug}\n"
        f"    cd data/kaggle/{slug} && for z in *.zip; do unzip -o $z; done\n\n"
        f"Or point $KAGGLE_DATA_DIR at a directory containing {slug}/.\n"
    )


def competition_path(slug: str, filename: str) -> Path:
    """Return a readable path for `filename` inside `slug`.

    Kaggle serves several competitions as `train.csv.zip` rather than
    `train.csv`, and what a mounted competition directory contains depends on
    whether Kaggle expanded the archive. So: try the plain name, then the same
    name in any sibling archive, then any file whose stem matches. The archive
    member is extracted once into a cache directory and that path is returned,
    so callers keep writing `pd.read_csv(competition_path(...))` with no zip
    handling of their own.
    """
    d = competition_dir(slug)
    direct = d / filename
    if direct.exists():
        return direct

    for f in sorted(d.iterdir()):
        if f.name.lower() == filename.lower():
            return f

    # Inside an archive?
    import zipfile

    for archive in sorted(d.glob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                hit = next((n for n in names
                            if Path(n).name == filename
                            or Path(n).name.lower() == filename.lower()), None)
                if hit is None:
                    hit = next((n for n in names
                                if Path(n).stem.lower() == Path(filename).stem.lower()), None)
                if hit is not None:
                    cache = Path(os.environ.get("KAGGLE_CACHE_DIR", "/tmp/kaggle-extract")) / slug
                    cache.mkdir(parents=True, exist_ok=True)
                    out = cache / Path(hit).name
                    if not out.exists() or out.stat().st_size == 0:
                        with zf.open(hit) as src, open(out, "wb") as dst:
                            dst.write(src.read())
                    return out
        except zipfile.BadZipFile:
            continue

    available = ", ".join(sorted(p.name for p in d.iterdir())[:20])
    raise CompetitionDataMissing(
        f"\n{filename!r} not found in {d}\nFiles present: {available}\n"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys
    for slug in sys.argv[1:] or ["forest-cover-type-kernels-only"]:
        try:
            print(f"{slug}: {competition_dir(slug)}")
        except CompetitionDataMissing as e:
            print(str(e).split("\n\nLooked")[0])
