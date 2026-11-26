#!/usr/bin/env python3
"""Download a REAL labelled dataset and write it as real_urls.csv.

  phishing  : OpenPhish public feed (https://openphish.com/feed.txt), no login needed
  legitimate: Tranco top-sites list (https://tranco-list.eu), domains only

Run:  python build_dataset.py            then:  python cli.py train --csv real_urls.csv

Fairness note: Tranco gives only domain names (no paths), while phishing links have paths. If we kept
the paths, the model could "cheat" by learning that a path means phishing. So by default BOTH sides
are cut down to scheme + host. Use --keep-paths only if you have real legitimate URLs with paths.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
import zipfile
from urllib.parse import urlsplit

OPENPHISH = "https://openphish.com/feed.txt"
TRANCO = "https://tranco-list.eu/top-1m.csv.zip"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "phish-detector-student-project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def host_only(url: str) -> str:
    p = urlsplit(url if "://" in url else "http://" + url)
    return f"{p.scheme}://{p.netloc}"


def build(phish_lines, tranco_rows, keep_paths=False, limit=5000, seed=1):
    rng = random.Random(seed)
    phish = [u.strip() for u in phish_lines if u.strip().startswith("http")]
    phish = list(dict.fromkeys(phish if keep_paths else (host_only(u) for u in phish)))
    rng.shuffle(phish)
    phish = phish[:limit]
    legit = [f"https://{d}" for _, d in tranco_rows if d]
    n = len(phish)                                   # same count of each, so the data is balanced
    legit = rng.sample(legit[:100000], min(n, len(legit[:100000])))
    return [(u, 1) for u in phish] + [(u, 0) for u in legit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real_urls.csv")
    ap.add_argument("--keep-paths", action="store_true")
    ap.add_argument("--limit", type=int, default=5000)
    a = ap.parse_args()
    print("Downloading OpenPhish feed ...")
    phish_lines = fetch(OPENPHISH).decode("utf-8", "replace").splitlines()
    print("Downloading Tranco list ...")
    z = zipfile.ZipFile(io.BytesIO(fetch(TRANCO)))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")))
    data = build(phish_lines, rows, a.keep_paths, a.limit)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "label"])
        w.writerows(data)
    bad = sum(l for _, l in data)
    print(f"Wrote {len(data)} rows to {a.out} ({bad} phishing, {len(data) - bad} legitimate).")
    print("Now run:  python cli.py train --csv " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
