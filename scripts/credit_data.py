#!/usr/bin/env python3
"""Shared loader for the retrospective credit-approval notebooks.

The raw file has no header, 16 columns, and `?` as a missing-value sentinel in
seven of them. The column semantics are not documented anywhere in the
repository, so they are inferred from value structure and stated explicitly
rather than assumed from the familiar UCI "default of credit card clients"
dataset — which this is *not*, despite the filename.

Keeping the loader in one place means all three columns of this topic parse the
data identically, so differences between them are differences in method and not
in parsing.
"""
from __future__ import annotations

import pandas as pd

DATA = "data/machine-learning/cc_approvals.data"

# Inferred schema. `kind` is "num" (continuous) or "cat" (nominal code).
# The names are descriptive placeholders: the upstream documentation for this
# file was not available, and inventing authoritative-sounding names for
# anonymised codes would misrepresent what is known.
COLUMNS = {
    "target":     "cat",  # 'a' = positive class (210), 'b' = negative (468)
    "limit_bal":  "num",  # 13.75 - 80.25, continuous credit limit
    "age_band":   "num",  # 0 - 28, continuous-coded age band
    "education":  "cat",  # {l, u, y}
    "marital":    "cat",  # {g, gg, p}
    "pay_status_0": "cat",  # 15 letter codes, ordinal repayment status
    "pay_status_2": "cat",  # 10 letter codes
    "pay_amt_3":  "num",  # 0 - 28.5, repayment amount ratio
    "bill_flag_4": "cat",  # {t, f}
    "bill_flag_5": "cat",  # {t, f}
    "past_due_6": "cat",   # zero-padded 2-digit codes, 0 - 40
    "bill_flag_7": "cat",  # {f, t}
    "pay_8":     "cat",  # {g, s, p}
    "bill_amt_1": "num",  # zero-padded, 0 - 20200
    "paid_amt_1": "num",  # 0 - 31285
    "default_next_month": "cat",  # '+' / '-'  <- the classification target
}

NUMERIC = [c for c, k in COLUMNS.items() if k == "num"]
CATEGORICAL = [c for c, k in COLUMNS.items() if k == "cat"]

# `target` and `default_next_month` are the label and its raw form. They are
# read for integrity checks but are never model features — including them would
# make the task trivially solvable and is the kind of mistake this notebook
# exists to rule out.
NON_FEATURES = ["target", "default_next_month"]
FEATURES = [c for c in COLUMNS if c not in NON_FEATURES]
FEATURE_NUMERIC = [c for c in NUMERIC if c not in NON_FEATURES]
FEATURE_CATEGORICAL = [c for c in CATEGORICAL if c not in NON_FEATURES]

MISSING_SENTINEL = "?"


def load_raw() -> pd.DataFrame:
    """Read the file with no header and no dtype coercion, as strings.

    Everything is read as text so that the `?` sentinel survives and the
    missingness is visible rather than silently turned into NaN by a
    converter.
    """
    return pd.read_csv(DATA, header=None, dtype=str, names=list(COLUMNS))


def load_clean(drop_rows_with_missing_target: bool = True) -> pd.DataFrame:
    """Parse into typed columns, replacing `?` with pandas NA.

    Numeric columns are coerced; the coercion is checked rather than assumed,
    because a column that fails to convert silently becomes all-NaN and drops
    out of the model without anyone noticing.
    """
    df = load_raw()

    if drop_rows_with_missing_target:
        df = df[df["default_next_month"] != MISSING_SENTINEL].copy()

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col].replace(MISSING_SENTINEL, pd.NA), errors="coerce")

    for col in CATEGORICAL:
        df[col] = df[col].replace(MISSING_SENTINEL, pd.NA)

    # Past-due codes are zero-padded strings ('01', '06'); they are ordinal
    # months, so convert to int to let the model treat them as ordered.
    df["past_due_6"] = pd.to_numeric(df["past_due_6"], errors="coerce")

    # Guard against a column that silently became all-NaN.
    for col in NUMERIC:
        if df[col].isna().all():
            raise ValueError(f"column {col!r} failed to convert — it would drop silently")

    df = df.reset_index(drop=True)
    df["y"] = (df["default_next_month"] == "+").astype(int)
    return df


def describe_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, for the EDA cells in column 1."""
    rows = []
    for col, kind in COLUMNS.items():
        if col in ("target", "default_next_month"):
            continue
        n_missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "kind": kind,
            "n_missing": n_missing,
            "pct_missing": 100 * n_missing / len(df),
            "n_unique": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False)
