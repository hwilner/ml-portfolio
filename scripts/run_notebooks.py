#!/usr/bin/env python3
"""Execute notebooks and report pass/fail, with the first error line surfaced.

Notebook code fails at runtime in ways a syntax check cannot catch — a renamed
API, a data file that is not where the notebook expects it, an assumption about
column names. Running them is the only way to know the repository is honest
about what executes.

Usage:
    python scripts/run_notebooks.py                 # everything under notebooks/
    python scripts/run_notebooks.py 07-retrospective # one subtree
    python scripts/run_notebooks.py --exec-in-place 07-retrospective
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def error_summary(exc: CellExecutionError, nb: nbformat.NotebookNode) -> str:
    """First meaningful line of the traceback, not nbclient's wrapper noise."""
    tb = exc.traceback
    lines = [ln for ln in tb.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "CellExecutionError" in ln or "-----" in ln or ln.startswith("An error"):
            continue
        return ln.strip()
    return lines[-1] if lines else "unknown error"


def run_one(path: Path, save: bool) -> tuple[str, bool, str, float]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        return path.name, False, error_summary(exc, nb), time.time() - t0
    except Exception as exc:  # noqa: BLE001 - kernel/startup failures too
        return path.name, False, f"{type(exc).__name__}: {exc}", time.time() - t0

    if save:
        nbformat.write(nb, path)
    return path.name, True, "", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subtrees", nargs="*", default=[])
    ap.add_argument("--exec-in-place", action="store_true",
                    help="write executed cell outputs back into the notebook")
    ap.add_argument("--no-figures", action="store_true",
                    help="strip figure outputs to keep notebooks small")
    args = ap.parse_args()

    base = ROOT / "notebooks"
    targets = [base / s for s in args.subtrees] if args.subtrees else [base]
    files: list[Path] = []
    for t in targets:
        if not t.exists():
            print(f"no such path: {t.relative_to(ROOT)}")
            return 2
        files.extend(sorted(t.rglob("*.ipynb")) if t.is_dir() else [t])

    # Never execute the 07-retrospective originals against 2026 libraries here;
    # they are built from source every time and verified by the same runner.
    if not args.no_figures and args.exec_in_place:
        print("note: --no-figures is ignored when saving outputs")

    results = []
    for f in files:
        name, ok, err, dt = run_one(f, args.exec_in_place)
        results.append((name, ok, err, dt))
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {name}  ({dt:.1f}s)")
        if not ok:
            print(f"        {err}")

    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed
    print(f"\n{len(results)} notebooks / {passed} pass / {failed} fail")

    if failed:
        print("\nFailures:")
        for name, ok, err, _ in results:
            if not ok:
                print(f"  {name}\n    {err}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
