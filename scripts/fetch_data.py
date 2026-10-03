"""Download the datasets that are too large to store in git.

Small datasets are committed under ``data/`` so the repository is usable
immediately after cloning. Anything above roughly 1 MB is fetched here instead,
which keeps ``git clone`` fast.

This script exists partly to replace a broken state: the repository previously
contained ``papers.csv.gz.part_aa`` through ``.part_ah`` - a 110 MB multi-part
download that was never reassembled, so the corresponding notebook could not
run. Those fragments are gone.

Every URL below was verified to return HTTP 200. Datasets with no usable raw
mirror are listed in :data:`MANUAL_DOWNLOADS` with instructions instead of a
link that would rot.

Usage::

    python scripts/fetch_data.py              # download everything available
    python scripts/fetch_data.py --list       # show available groups
    python scripts/fetch_data.py --check      # report status, download nothing
    python scripts/fetch_data.py --only lda   # fetch one group

Example:
    >>> from fetch_data import DATASETS, MANUAL_DOWNLOADS
    >>> sorted({d.key for d in DATASETS})
    ['song-genres']
    >>> sorted({d.key for d in MANUAL_DOWNLOADS})
    ['asl-letters', 'bees-images', 'lda']
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# 200 MB is larger than most CI timeouts, so per-read timeout is generous and
# failures are reported rather than raised.
_TIMEOUT_SECONDS = 180
_CHUNK_BYTES = 1 << 20


class Dataset(NamedTuple):
    """One downloadable artefact.

    Attributes:
        key: Short identifier used with ``--only``.
        destination: Path relative to the repository root.
        urls: Candidate URLs, tried in order until one succeeds.
        size_mb: Approximate size, for reporting only.
        note: Why the file is not committed to git.
    """

    key: str
    destination: str
    urls: tuple[str, ...]
    size_mb: float
    note: str


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="song-genres",
        destination="data/machine-learning/fma-rock-vs-hiphop.csv",
        urls=(
            "https://raw.githubusercontent.com/shukkkur/"
            "Classify-Song-Genres-from-Audio-Data/main/datasets/fma-rock-vs-hiphop.csv",
        ),
        size_mb=3.3,
        note="Track metadata with genre labels, from the Free Music Archive.",
    ),
    Dataset(
        key="song-genres",
        destination="data/machine-learning/echonest-metrics.json",
        urls=(
            "https://raw.githubusercontent.com/shukkkur/"
            "Classify-Song-Genres-from-Audio-Data/main/datasets/echonest-metrics.json",
        ),
        size_mb=2.1,
        note="Eight pre-extracted audio features per track (echonest).",
    ),
)


class ManualDownload(NamedTuple):
    """A dataset that has no usable raw mirror and must be fetched by hand.

    Attributes:
        key: Group name, matching the ``--only`` flag.
        destination: Where the file should end up.
        size_mb: Approximate size.
        instructions: Concrete steps a reader can follow.
    """

    key: str
    destination: str
    size_mb: float
    instructions: str


MANUAL_DOWNLOADS: tuple[ManualDownload, ...] = (
    ManualDownload(
        key="lda",
        destination="data/nlp/papers.csv.gz",
        size_mb=200.0,
        instructions=(
            "The NIPS paper corpus is ~200 MB, which exceeds GitHub's raw-file "
            "limit, so there is no stable direct link. Download it from Kaggle:\n"
            "    https://www.kaggle.com/datasets/benhamner/nips-papers\n"
            "then gunzip papers.csv and save it to data/nlp/papers.csv.gz.\n"
            "The LDA notebook runs without it, using a synthetic stand-in that "
            "is labelled as such on every figure."
        ),
    ),
    ManualDownload(
        key="bees-images",
        destination="data/computer-vision/naive-bees/images.zip",
        size_mb=3.3,
        instructions=(
            "Bee photographs from the DataCamp 'Naive Bees' project. No raw "
            "GitHub mirror is reliable for this file. Ask the project owner to "
            "add it, or substitute any folder of labelled images.\n"
            "Without it the notebooks generate a clearly-labelled synthetic "
            "stand-in, and every score they print describes synthetic shapes "
            "rather than photographs."
        ),
    ),
    ManualDownload(
        key="asl-letters",
        destination="data/computer-vision/asl-letters/",
        size_mb=26.0,
        instructions=(
            "ASL letter photographs, from the Udacity Sign Language "
            "Recognizer project:\n"
            "    https://s3.amazonaws.com/udacitysaints/data/train.zip\n"
            "Unzip into data/computer-vision/asl-letters/, giving subfolders "
            "A/, B/ and C/."
        ),
    ),
)


def available() -> list[Dataset]:
    """Return the datasets not yet present on disk.

    Returns:
        list[Dataset]: Entries whose destination does not exist.
    """
    return [d for d in DATASETS if not (REPO_ROOT / d.destination).exists()]


def _download(dataset: Dataset) -> bool:
    """Fetch one dataset from the first URL that responds.

    Args:
        dataset: The artefact to fetch.

    Returns:
        bool: ``True`` on success, ``False`` if every candidate URL failed.
    """
    target = REPO_ROOT / dataset.destination
    target.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in dataset.urls:
        try:
            print(f"  fetching {dataset.destination} ({dataset.size_mb:.1f} MB)")
            with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
                with target.open("wb") as handle:
                    # Stream in chunks so a multi-MB file never lands in memory.
                    while chunk := response.read(_CHUNK_BYTES):
                        handle.write(chunk)
            size_mb = target.stat().st_size / 1e6
            print(f"    -> {target.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"    failed: {exc}")

    print(f"  could not fetch {dataset.destination}: {last_error}")
    return False


def main(argv: list[str] | None = None) -> int:
    """Run the fetch script.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        int: ``0`` on success, ``1`` if one or more downloads failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--list", action="store_true", help="list dataset groups and exit"
    )
    parser.add_argument(
        "--check", action="store_true", help="report which files are missing"
    )
    parser.add_argument(
        "--only",
        metavar="KEY",
        help="fetch only datasets matching this key",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("automatic downloads:")
        for dataset in DATASETS:
            state = "present" if (REPO_ROOT / dataset.destination).exists() else "missing"
            print(f"  {dataset.key:14} [{state:7}] {dataset.destination}")
        print("\nmanual downloads (no stable direct link):")
        for manual in MANUAL_DOWNLOADS:
            print(f"  {manual.key:14} {manual.destination}")
            for line in manual.instructions.splitlines():
                print(f"      {line}")
        return 0

    selected = DATASETS
    if args.only:
        selected = tuple(d for d in DATASETS if d.key == args.only)
        if not selected:
            for manual in MANUAL_DOWNLOADS:
                if manual.key == args.only:
                    print(f"'{args.only}' must be downloaded manually:\n")
                    print(manual.instructions)
                    return 0
            print(f"No dataset matches key {args.only!r}.", file=sys.stderr)
            return 1

    missing = [d for d in selected if not (REPO_ROOT / d.destination).exists()]

    if args.check:
        print(f"DATA_DIR: {DATA_DIR}\n")
        for dataset in selected:
            state = "MISSING" if dataset in missing else "present"
            print(f"  {state:8} {dataset.destination}")
        return 0

    if not missing:
        print("Nothing to do: all selected datasets are already present.")
        return 0

    print(f"Fetching {len(missing)} dataset(s) into {DATA_DIR}\n")
    failures = sum(1 for d in missing if not _download(d))
    if failures:
        print(f"\n{failures} download(s) failed - see messages above.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
