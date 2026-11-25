"""Load a real labelled URL list from a CSV file."""
from __future__ import annotations

import csv

URL_NAMES = ("url", "urls", "link", "website")
LABEL_NAMES = ("label", "labels", "class", "type", "status", "result")
BAD = {"1", "bad", "phishing", "phish", "malicious", "malware", "defacement", "spam", "yes", "true"}
GOOD = {"0", "good", "benign", "legitimate", "legit", "safe", "no", "false", "-1"}


def load_csv(path: str, url_col: str | None = None, label_col: str | None = None) -> list[tuple[str, int]]:
    """Read ``(url, label)`` pairs; label 1 = phishing/malicious, 0 = legitimate.

    Column names are guessed (url / label / class / type ...) unless you pass them.
    Rows whose label is not recognised are skipped, and the count is reported by the caller.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        u = cols.get((url_col or "").lower()) or next((cols[n] for n in URL_NAMES if n in cols), None)
        lab = cols.get((label_col or "").lower()) or next((cols[n] for n in LABEL_NAMES if n in cols), None)
        if not u or not lab:
            raise ValueError(f"Could not find URL and label columns in {list(cols.values())}. "
                             "Use --url-col and --label-col.")
        rows, seen = [], set()
        for r in reader:
            url, raw = (r.get(u) or "").strip(), (r.get(lab) or "").strip().lower()
            if not url or url in seen:
                continue
            if raw in BAD:
                rows.append((url, 1))
            elif raw in GOOD:
                rows.append((url, 0))
            else:
                continue
            seen.add(url)
    if not rows:
        raise ValueError("No usable rows found (check the label values).")
    return rows
