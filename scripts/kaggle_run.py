#!/usr/bin/env python3
"""Run a repository notebook on Kaggle's own hardware, then bring the results back.

The sandbox this repository is developed in has 2 CPUs, 3 GB of RAM and no GPU.
Kaggle gives a kernel two Tesla T4s (14.6 GiB each) and ~30 GB of RAM, and the
Kaggle API can push, poll and download a kernel without a browser. So a notebook
that cannot execute here can still be executed, for real, on Kaggle — which is
the difference between a notebook that runs and a notebook that is claimed to run.

What this does, per notebook:

  1. writes a kernel directory next to the notebook, with the converted
     `kernel-metadata.json` and the notebook as the code file
  2. `kaggle kernels push` uploads it and starts a run
  3. polls `kernels status` until the run completes or fails
  4. on success, downloads two things:
       - `results.json`, which the notebook itself writes, and
       - the executed notebook, which comes back WITH its outputs, so the
         committed `.ipynb` has real cells and real results in it

If the pull returns a notebook without outputs (Kaggle's behaviour varies with
kernel settings) the results are still available from `results.json` and from
the run log, and this script says so rather than silently reporting success.

Usage:
    python scripts/kaggle_run.py 08-kaggle/01-forest-cover-type/03-col-3-2026-tools.py
    python scripts/kaggle_run.py --gpu 08-kaggle/04-gemma-finetune/...
    python scripts/kaggle_run.py --dry-run <notebook.py>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = "/workspace/.venv/bin/kaggle"
KAGGLE_ENV = {"KAGGLE_CONFIG_DIR": "/workspace/.kaggle"}

# Competition and dataset slugs a kernel needs mounted, keyed by notebook path.
# Anything not listed runs with no attached data, which is correct for the
# notebooks that generate their own.
ATTACHMENTS: dict[str, dict] = {
    "01-forest-cover-type": {
        "competition_sources": ["forest-cover-type-kernels-only"],
    },
    "02-dont-overfit": {
        "competition_sources": ["dont-overfit-ii"],
    },
    "03-homesite-quote": {
        "competition_sources": ["homesite-quote-conversion"],
    },
    "04-optiver-lgbm": {
        "competition_sources": ["optiver-trading-at-the-close"],
    },
    "05-mistral-rag": {
        "dataset_sources": ["meruvulikith/1300-towards-datascience-medium-articles-dataset"],
    },
}


def kaggle(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        [KAGGLE, *args], capture_output=True, text=True,
        env={**__import__("os").environ, **KAGGLE_ENV}, timeout=1800,
    )
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def topic_for(path: Path) -> str:
    return path.parent.name


def build_kernel_dir(notebook: Path, kernel_id: str, gpu: bool) -> Path:
    """Create a pushable kernel directory containing the notebook."""
    workdir = ROOT / ".kaggle-kernels" / notebook.stem
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    # Small repository data the kernel genuinely needs. `data/kaggle` is
    # excluded on purpose: those files mount from competition_sources at
    # /kaggle/input, and copying them would push 447 MB of data Kaggle is
    # already providing.
    IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.zip", "kaggle",
                                    "bee_imgs", "nips", "nytkids", "*.png")
    src = ROOT / "data"
    if src.is_dir():
        shutil.copytree(src, workdir / "data", ignore=IGNORED,
                        ignore_dangling_symlinks=True)

    import nbformat as nbf
    from nbclient import NotebookClient

    nb = nbf.read(notebook, as_version=4)

    # Execute nothing here: the run happens on Kaggle. Clearing outputs keeps
    # the pushed payload small and makes it unambiguous which outputs in the
    # committed notebook came from Kaggle rather than from this machine.
    # Only code cells carry `outputs`. Assigning it to a markdown cell adds a
    # property that nbformat forbids, and Kaggle's validator rejects the whole
    # notebook with "Additional properties are not allowed ('outputs')" — which
    # surfaces as a papermill conversion error rather than as a schema error.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, workdir / "main.ipynb")

    attach = dict(ATTACHMENTS.get(topic_for(notebook), {}))
    # Kaggle refuses (409) a kernel whose title does not slugify back to its id,
    # so the title is derived from the id rather than from the filename.
    slug = kernel_id.split("/", 1)[-1]
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " "),
        "code_file": "main.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "enable_gpu": gpu,
        "enable_internet": True,
        **attach,
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return workdir


def wait_for(kernel_id: str, poll: int = 45, max_minutes: int = 180) -> str:
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        status = kaggle("kernels", "status", kernel_id, check=False).strip()
        tag = status.split("status")[-1].strip().strip('"') if "status" in status else status
        print(f"    {tag}", flush=True)
        if "COMPLETE" in tag or "ERROR" in tag or "FAIL" in tag:
            return tag
        time.sleep(poll)
    return "TIMEOUT"


def read_log(outdir: Path) -> str:
    """Return the kernel's stdout.

    `kaggle kernels output` returns a run log that is a single JSON array of
    stream records, not newline-delimited objects — reading it line by line
    yields nothing, which is a quiet way to lose every result. And
    `kaggle kernels pull` returns the SOURCE notebook: a successful run comes
    back with zero outputs. So the log is the only reliable evidence channel,
    and a notebook that needs its numbers to survive the round trip should
    also write a results.json into its working directory.
    """
    import json as _json
    for f in sorted(outdir.glob("*.log")):
        try:
            recs = _json.loads(f.read_text(errors="replace"))
        except Exception:
            continue
        if isinstance(recs, list):
            return "".join(r.get("data", "") for r in recs
                           if isinstance(r, dict) and r.get("stream_name") == "stdout")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", help="path to the .ipynb to run on Kaggle")
    ap.add_argument("--kernel-id", default=None)
    ap.add_argument("--gpu", action="store_true", help="request a GPU (2x Tesla T4)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the kernel dir and stop")
    ap.add_argument("--max-minutes", type=int, default=180)
    args = ap.parse_args()

    notebook = Path(args.notebook)
    if not notebook.is_absolute():
        notebook = ROOT / "notebooks" / args.notebook
    if not notebook.exists():
        raise SystemExit(f"no such notebook: {notebook}")

    gpu = args.gpu and not args.no_gpu
    kernel_id = args.kernel_id or f"hareljwil/portfolio-{notebook.stem[:40]}"

    workdir = build_kernel_dir(notebook, kernel_id, gpu)
    print(f"kernel dir : {workdir.relative_to(ROOT)}")
    print(f"kernel id  : {kernel_id}")
    print(f"gpu        : {gpu}")
    if args.dry_run:
        return 0

    print("\npushing ...", flush=True)
    kaggle("kernels", "push", "-p", str(workdir))

    print(f"\nwaiting (max {args.max_minutes} min) ...", flush=True)
    status = wait_for(kernel_id, max_minutes=args.max_minutes)
    print(f"\nstatus: {status}")

    if "COMPLETE" not in status:
        print("\nthe run did not complete; fetching the log for diagnosis")
        log = kaggle("kernels", "output", kernel_id, "-p", str(workdir / "log"), check=False)
        for f in sorted((workdir / "log").glob("*")) if (workdir / "log").is_dir() else []:
            print(f"\n--- {f.name} (last 60 lines) ---")
            print("\n".join(f.read_text(errors="replace").splitlines()[-60:]))
        return 1

    # Bring back the executed notebook and the results file.
    outdir = workdir / "output"
    kaggle("kernels", "output", kernel_id, "-p", str(outdir), check=False)
    stdout_text = ""
    if outdir.is_dir():
        for f in sorted(outdir.iterdir()):
            print(f"  output: {f.name}  ({f.stat().st_size:,} bytes)")
        results = outdir / "results.json"
        if results.exists():
            print("\nresults.json:")
            print(results.read_text()[:4000])
        stdout_text = read_log(outdir)

    if stdout_text.strip():
        (workdir / "stdout.txt").write_text(stdout_text)
        print(f"\n--- stdout from Kaggle ({len(stdout_text):,} chars) ---")
        print(stdout_text[:3000])

    back = workdir / "executed"
    kaggle("kernels", "pull", kernel_id, "-p", str(back), check=False)
    nb_back = back / "main.ipynb"
    if nb_back.exists():
        import nbformat as nbf
        nb = nbf.read(nb_back, as_version=4)
        with_outputs = sum(
            1 for c in nb.cells for o in c.get("outputs", [])
            if o.output_type in ("execute_result", "display_data", "stream")
        )
        print(f"\nexecuted notebook: {len(nb.cells)} cells, "
              f"{with_outputs} outputs captured from Kaggle")
        if with_outputs:
            nbf.write(nb, notebook)
            print(f"  committed to {notebook.relative_to(ROOT)}")
        else:
            print("  NOTE: Kaggle returned the notebook WITHOUT outputs. The run")
            print("        succeeded — results.json above is the evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
