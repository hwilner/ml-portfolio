#!/usr/bin/env python3
"""Build the retrospective notebooks from source cells.

A .ipynb is JSON, and hand-writing JSON means hand-escaping every quote and
newline. Writing the cells as plain Python source and compiling them through
this builder keeps the source readable and guarantees the notebook is valid
JSON that nbformat can execute.

Usage:
    python scripts/build_notebook.py                    # build all
    python scripts/build_notebook.py 01-causal-inference # build one topic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_ROOT = ROOT / "notebooks"

# Notebooks that import a repository helper get that helper's source inlined
# into the built .ipynb as a first code cell.
#
# Why: `kaggle kernels push` uploads ONLY the notebook. A probe kernel pushed
# with a `scripts/` directory alongside it came up with /kaggle/working holding
# nothing but __notebook__.ipynb, so `from scripts.kaggle_data import ...` dies
# on Kaggle even though the package sits right next to the notebook locally.
# Inlining keeps one source of truth for the logic while making every built
# notebook self-contained on GitHub, on Kaggle, and in CI.
INLINE_MODULES = {
    "scripts.kaggle_data": ROOT / "scripts" / "kaggle_data.py",
    "scripts.credit_data": ROOT / "scripts" / "credit_data.py",
}

# The builder walks every topic directory under notebooks/. The two retrospective
# trees (07-retrospective, 08-kaggle) use the same source layout, so there is no
# reason for the path to be hardcoded to one of them.
TREES = [d for d in (NB_ROOT / "07-retrospective", NB_ROOT / "08-kaggle") if d.is_dir()]


def topics() -> list[str]:
    out: list[str] = []
    for tree in TREES:
        out += [d.name for d in sorted(tree.iterdir()) if d.is_dir()]
    return out


def resolve(name: str) -> Path | None:
    for tree in TREES:
        cand = tree / name
        if cand.is_dir():
            return cand
    return None


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(directory: Path) -> Path | None:
    """Compile `<name>/source.py` into `<name>/<name>.ipynb`."""
    py_files = sorted(p for p in directory.glob("*.py"))
    if not py_files:
        return None

    for py_file in py_files:
        name = py_file.stem
        out = py_file.with_suffix(".ipynb")
        source = py_file.read_text(encoding="utf-8")

        # Each top-level `# %% [markdown]` / `# %%` marker starts a new cell.
        cells: list[nbf.NotebookNode] = []
        kind, buf = "md", []

        def flush() -> None:
            text = "\n".join(buf).strip("\n")
            if text:
                cells.append(md(text) if kind == "md" else code(text))

        for line in source.splitlines():
            if line.startswith("# %%"):
                flush()
                kind = "md" if "[markdown]" in line else "code"
                buf = []
            else:
                buf.append(line)
        flush()

        # Inline any repository helper the notebook imports, then drop the
        # import statement itself so the cell does not re-trigger it.
        # `from scripts.kaggle_data import competition_path` does NOT contain
        # the substring "import scripts.kaggle_data", so both spellings have to
        # be tested. Getting this wrong silently skips the inlining and the
        # notebook fails on Kaggle with a bare ModuleNotFoundError.
        def imports(cell_source: str, mod: str) -> bool:
            return f"from {mod} import" in cell_source or f"import {mod}" in cell_source

        needed = [mod for mod in INLINE_MODULES
                  if any(imports(c.source, mod) for c in cells
                         if c.cell_type == "code")]
        if needed:
            parts = []
            for mod in needed:
                path = INLINE_MODULES[mod]
                src = path.read_text(encoding="utf-8").strip("\n")
                parts.append(f"# ---- inlined from {path.relative_to(ROOT)} "
                             f"({mod}) ----\n{src}")
            parts.append(
                "# `scripts` is inlined above rather than imported, because a "
                "Kaggle kernel\n"
                "# receives only the notebook file. See scripts/build_notebook.py."
            )
            cells.insert(0, code("\n\n\n".join(parts)))
            for c in cells:
                if c.cell_type == "code":
                    for mod in needed:
                        c.source = c.source.replace(f"from {mod} import "
                                                   f"competition_path, competition_dir\n", "")
                        c.source = c.source.replace(f"from {mod} import "
                                                   f"competition_path\n", "")
                        c.source = c.source.replace(f"from {mod} import "
                                                   f"competition_dir\n", "")

        # A syntax error in one cell surfaces as a confusing failure much later
        # during execution, so it is worth catching here at build time.
        for i, cell in enumerate(cells):
            if cell.cell_type != "code":
                continue
            try:
                compile(cell.source, f"{name}:cell{i}", "exec")
            except SyntaxError as exc:
                raise SystemExit(
                    f"\n  SYNTAX ERROR in {py_file.name} cell {i} "
                    f"(line {exc.lineno}): {exc.msg}\n  {exc.text or ''}"
                ) from exc

        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata.update(
            {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            }
        )
        nbf.validate(nb)
        nbf.write(nb, out)
        print(f"  {out.relative_to(ROOT)}  ({len(cells)} cells)")

    return directory


def main() -> int:
    targets = sys.argv[1:] or topics()
    built = 0
    for name in targets:
        directory = resolve(name)
        if directory is None:
            print(f"  ! no such topic: {name}")
            continue
        build(directory)
        built += 1
    if not built:
        print("nothing to build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
