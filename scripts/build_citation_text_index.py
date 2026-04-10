#!/usr/bin/env python3
"""Build a citation -> text lookup pickle from laws_de.csv + court_considerations.csv.

For courts: a single citation can have multiple text rows (different parts of
one consideration). Join them with a space delimiter and de-dup.

Output: artifacts/citation_text_index.pkl
  Format: dict[str, str]  (citation -> snippet text, truncated to 1500 chars)
"""

from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAWS = REPO / "data/laws_de.csv"
COURT = REPO / "data/court_considerations.csv"
OUT = REPO / "artifacts/citation_text_index.pkl"

MAX_CHARS_PER_CITE = 1500  # cap to control prompt token cost


def load_laws() -> dict[str, str]:
    out: dict[str, str] = {}
    with LAWS.open(newline="") as f:
        # Bump csv field size in case of huge fields
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(f)
        for row in reader:
            cite = (row.get("citation") or "").strip()
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if not cite:
                continue
            full = f"{title}\n{text}" if title else text
            full = full.replace("\n", " ").strip()
            if len(full) > MAX_CHARS_PER_CITE:
                full = full[: MAX_CHARS_PER_CITE - 1] + "…"
            # First occurrence wins (laws_de should be unique by cite anyway)
            if cite not in out:
                out[cite] = full
    return out


def load_courts() -> dict[str, str]:
    out: dict[str, list[str]] = {}
    with COURT.open(newline="") as f:
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(f)
        for row in reader:
            cite = (row.get("citation") or "").strip()
            text = (row.get("text") or "").strip()
            if not cite or not text:
                continue
            out.setdefault(cite, []).append(text)
    final: dict[str, str] = {}
    for cite, parts in out.items():
        joined = " ".join(parts).replace("\n", " ").strip()
        if len(joined) > MAX_CHARS_PER_CITE:
            joined = joined[: MAX_CHARS_PER_CITE - 1] + "…"
        final[cite] = joined
    return final


def main():
    if OUT.exists():
        print(f"{OUT} already exists. Delete it to rebuild.")
        return

    t0 = time.time()
    print("Loading laws_de.csv ...", flush=True)
    laws = load_laws()
    print(f"  {len(laws):,} unique law citations  ({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    print("Loading court_considerations.csv ...", flush=True)
    courts = load_courts()
    print(f"  {len(courts):,} unique court citations  ({time.time()-t1:.1f}s)", flush=True)

    combined = {}
    combined.update(laws)
    combined.update(courts)
    print(f"Combined unique citations: {len(combined):,}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        pickle.dump(combined, f)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
