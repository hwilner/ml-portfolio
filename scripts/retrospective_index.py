#!/usr/bin/env python3
"""Generate the notebook index for RETROSPECTIVE.md.

The index is built from the notebooks that actually exist and execute, rather
than maintained by hand, so it cannot drift from the repository.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRO = ROOT / "notebooks" / "07-retrospective"

TOPIC_TITLES = {
    "01-causal-inference": (
        "Regression discontinuity: bank debt recovery",
        "Causal inference, the strongest project in the repository",
    ),
    "02-tabular-fairness": (
        "Credit approval and fairness",
        "Tabular ML, and a lending decision examined for disparate impact",
    ),
    "03-audio-signal-processing": (
        "Song genre: computing the representation",
        "Where the project moves from applied statistics to signal processing",
    ),
    "04-nlp-topic-modelling": (
        "Topic modelling on children's bestsellers",
        "Bag-of-words versus embeddings, and the checks a topic model needs",
    ),
    "05-computer-vision": (
        "Naive bees: HOG, PCA and SVM",
        "A classical vision pipeline, reconstructed and verified",
    ),
    "06-deep-learning": (
        "Multilayer perceptron from scratch",
        "The project called the 2019 retrospective's biggest missed opportunity",
    ),
}

COLUMN_LABELS = {
    "01": "2019, as written",
    "02": "2019, with current judgement",
    "03": "2026 tools",
}


def first_heading(path: Path) -> str:
    """First H1-level heading in the notebook, used as its title.

    Most notebooks open with a level-2 "What this notebook is" section, so
    that heading is not distinctive; the `# Title` on the first line is.
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(r'"#\s+(?!#)([^"\n]+?)\\n"', text)
    if match:
        return match.group(1).strip()
    match = re.search(r'"##\s*(?!What this notebook)([^"\n]+?)\\n"', text)
    return match.group(1).strip() if match else path.stem


def build() -> str:
    lines = [
        "## The three columns, as notebooks",
        "",
        "The analysis above is prose. These are the same three columns as code,",
        "runnable, with every number below produced by a cell that executed.",
        "",
        "Each topic is a directory under `notebooks/07-retrospective/`, with one",
        "notebook per column. Source lives alongside as `.py` and is compiled by",
        "`scripts/build_notebook.py`; execution is verified by",
        "`scripts/run_notebooks.py`.",
        "",
    ]

    total = 0
    for name in sorted(TOPIC_TITLES):
        directory = RETRO / name
        notebooks = sorted(directory.glob("*.ipynb"))
        if not notebooks:
            continue
        title, blurb = TOPIC_TITLES[name]
        total += len(notebooks)

        lines.append(f"### {name} — {title}")
        lines.append("")
        lines.append(f"{blurb}")
        lines.append("")
        for nb in notebooks:
            col = nb.name[:2]
            lines.append(f"- `{nb.name}` — **{COLUMN_LABELS.get(col, col)}**  ")
            lines.append(f"  {first_heading(nb)}")
        lines.append("")

    lines.append(
        f"**{total} notebooks across {len(TOPIC_TITLES)} topics.** Every one "
        "executes, and `scripts/run_notebooks.py` fails the build if any does not."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(build())
