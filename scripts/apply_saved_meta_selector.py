#!/usr/bin/env python3
"""
Apply a saved meta-selector pickle to a judged-bundles JSON and write
a submission CSV. No random search, no retraining — pure inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from run_v11_meta_selector import (  # noqa: E402
    SelectorConfig,
    build_rows,
    group_rows_by_query,
    predict_rows,
    select_predictions,
    write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pkl", type=Path, required=True)
    parser.add_argument("--apply-judged", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.model_pkl.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    source_names = bundle["source_names"]
    cfg_dict = bundle["selector_config"]
    cfg = SelectorConfig(**cfg_dict)

    payload = json.loads(args.apply_judged.read_text())
    bundles = payload["bundles"]
    rows = build_rows(bundles, gold_map=None, source_names=source_names)
    predict_rows(model, rows)

    grouped = group_rows_by_query(rows)
    predictions = {
        qid: select_predictions(query_rows, cfg)
        for qid, query_rows in grouped.items()
    }

    write_predictions(args.output_csv, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
