# SQL

## International debt statistics

**File:** [`01-international-debt-statistics.ipynb`](01-international-debt-statistics.ipynb)
**Status:** ✅ runs
**Competencies:** SQL ●● · data analysis ●·

Analysing the World Bank's international debt dataset with PostgreSQL, driven
from Jupyter via the `%%sql` cell magic and `sqlalchemy`.

### What it covers

| Step | Technique |
|---|---|
| 1 | Connect and preview the table |
| 2 | `COUNT(DISTINCT country_name)` — dataset scope |
| 3 | `SELECT DISTINCT` on the indicator code — what is being measured |
| 4 | `SUM(debt)` scaled to millions, `ROUND` — global total |
| 5 | `GROUP BY` + `ORDER BY ... DESC LIMIT 1` — largest debtor |
| 6 | `AVG` grouped by indicator — which debt category dominates |
| 7 | Correlated subquery matching a row against `MAX` within a category |
| 8 | `COUNT` + `GROUP BY` + `ORDER BY` — most common indicator |
| 9 | Multi-column `GROUP BY` with `MAX` — final ranking |

The progression is deliberate: start with scope, then totals, then drill into
the category that dominates, then the individual record. That is a reasonable
order for exploratory analysis, and step 7's correlated subquery is the piece
most worth understanding — it finds the row holding a maximum without a
self-join.

Note the scale handling in steps 4 and 5: raw debt values run to eleven
significant figures, so they are divided into millions before display. Small
touch, but it is the difference between a readable result and a wall of digits.

### Reproducing

```bash
pip install sqlalchemy psycopg2-binary
unzip data/sql/international_debt.zip -d data/sql/
# adapt the connection cell to your local PostgreSQL instance
```

The notebook's first cell connects with `postgresql:///international_debt`,
which assumes a local database of that name. Change it to point elsewhere.
