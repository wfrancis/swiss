#!/usr/bin/env python3
"""
Inject procedural/boilerplate citations based on proceeding type.

Swiss court decisions always cite certain procedural provisions depending
on the type of proceeding (criminal appeal, civil appeal, social insurance, etc.).
These are invisible to FAISS/BM25 because they're topically unrelated to the query.

This script detects the proceeding type from the query + existing predictions,
then injects the standard procedural citations.

Usage:
    .venv/bin/python scripts/inject_procedural.py \
        --baseline submissions/val_pred_baseline_public_best_30681.csv \
        --output submissions/val_pred_procedural_inject.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Proceeding type detection: look for key statute references in predictions
# and query text to determine what kind of case this is.

PROCEEDING_RULES = {
    "criminal_bstger": {
        # Federal Criminal Court (Bundesstrafgericht) appeal
        "detect_in_predictions": ["StPO", "StGB"],
        "detect_in_query": [r"Bundesstrafgericht", r"StPO", r"Straf", r"angeklagt", r"beschuldigt",
                            r"Untersuchungshaft", r"detention", r"criminal", r"accused"],
        "inject": [
            "Art. 37 Abs. 1 StBOG",   # BStGer jurisdiction
            "Art. 39 Abs. 1 StBOG",   # BStGer procedure
            "Art. 422 Abs. 1 StPO",   # costs composition
            "Art. 422 Abs. 2 StPO",   # costs details
            "Art. 390 Abs. 2 StPO",   # appeal procedure
            "Art. 384 StPO",          # appeal deadline
            "Art. 385 Abs. 1 StPO",   # appeal grounds requirement
            "Art. 135 Abs. 4 StPO",   # legal aid repayment
            "Art. 396 Abs. 1 StPO",   # appeal scope
        ],
    },
    "criminal_sentencing": {
        "detect_in_predictions": ["StGB"],
        "detect_in_query": [r"Straf", r"Freiheitsstrafe", r"Geldstrafe", r"sentenc", r"punish",
                            r"conviction", r"Verurteil"],
        "inject": [
            "Art. 42 Abs. 1 StGB",    # suspended sentence
            "Art. 44 Abs. 1 StGB",    # probation period
            "Art. 49 Abs. 2 StGB",    # additional sentence
            "Art. 12 Abs. 1 StGB",    # intent requirement
            "Art. 25 StGB",           # aiding
            "Art. 26 StGB",           # special duty
            "Art. 333 Abs. 1 StGB",   # general provisions scope
        ],
    },
    "criminal_appeal_costs": {
        "detect_in_predictions": ["StPO"],
        "detect_in_query": [r"Beschwerde", r"appeal", r"Rechtsmittel", r"recours"],
        "inject": [
            "Art. 436 Abs. 1 StPO",   # appeal compensation
            "Art. 436 Abs. 2 StPO",   # partial acquittal costs
        ],
    },
    "civil_appeal_bgg": {
        "detect_in_predictions": ["BGG", "ZGB", "OR", "ZPO"],
        "detect_in_query": [r"Bundesgericht", r"BGG", r"Federal Supreme Court", r"appeal"],
        "inject": [
            "Art. 93 Abs. 1 BGG",     # interim decisions
            "Art. 113 BGG",           # constitutional complaint
        ],
    },
    "social_insurance": {
        "detect_in_predictions": ["ATSG", "IVG", "UVG", "KVG"],
        "detect_in_query": [r"Versicherung", r"insurance", r"IV", r"disability", r"Invalidität",
                            r"Rente", r"pension", r"SUVA"],
        "inject": [
            "Art. 56 Abs. 1 ATSG",    # appeal against decisions
            "Art. 60 Abs. 1 ATSG",    # appeal deadline
            "Art. 21 Abs. 4 ATSG",    # treatment refusal
        ],
    },
    "family_maintenance": {
        "detect_in_predictions": ["ZGB"],
        "detect_in_query": [r"Unterhalt", r"maintenance", r"child support", r"Kindes",
                            r"custody", r"Sorgerecht", r"Obhut"],
        "inject": [
            "Art. 285 Abs. 1 ZGB",    # child support criteria
            "Art. 288 Abs. 1 ZGB",    # lump-sum child support
            "Art. 308 Abs. 1 ZGB",    # child protection deputy
        ],
    },
    "inheritance": {
        "detect_in_predictions": ["ZGB"],
        "detect_in_query": [r"Erb", r"Testament", r"will", r"testat", r"Nachlass", r"inherit",
                            r"Erblasser", r"succession", r"Verfügung von Todes"],
        "inject": [
            "Art. 467 ZGB",           # testamentary capacity
            "Art. 505 Abs. 1 ZGB",    # holographic will requirements
            "Art. 519 Abs. 1 ZGB",    # will invalidity
            "Art. 520 Abs. 1 ZGB",    # formal defects
            "Art. 458 Abs. 3 ZGB",    # representation by issue
        ],
    },
    "contract_interpretation": {
        "detect_in_predictions": ["OR"],
        "detect_in_query": [r"Vertrag", r"contract", r"Auslegung", r"interpret", r"Willen",
                            r"Vereinbarung", r"agreement"],
        "inject": [
            "Art. 18 Abs. 1 OR",      # contract interpretation
            "Art. 20 Abs. 2 OR",      # partial nullity
            "Art. 100 Abs. 2 OR",     # liability waiver limits
        ],
    },
    "mandate_fiduciary": {
        "detect_in_predictions": ["OR"],
        "detect_in_query": [r"Auftrag", r"mandate", r"fiduc", r"Beauftrag", r"Sorgfalt",
                            r"investment", r"Anlage"],
        "inject": [
            "Art. 397 Abs. 1 OR",     # mandate instructions
            "Art. 4 ZGB",             # judicial discretion
        ],
    },
    "property_possession": {
        "detect_in_predictions": ["ZGB"],
        "detect_in_query": [r"Eigentum", r"property", r"Besitz", r"possess", r"Herausgabe",
                            r"Sache", r"Kulturgut", r"stolen"],
        "inject": [
            "Art. 641 Abs. 2 ZGB",    # right to reclaim property
            "Art. 934 Abs. 1bis ZGB", # cultural property recovery
            "Art. 940 Abs. 1 ZGB",    # bad faith possessor liability
        ],
    },
    "capacity": {
        "detect_in_predictions": ["ZGB"],
        "detect_in_query": [r"urteilsfähig", r"capacity", r"Handlungsfähig", r"mündig"],
        "inject": [
            "Art. 16 ZGB",            # capacity definition
        ],
    },
    "civil_procedure_general": {
        "detect_in_predictions": ["ZPO"],
        "detect_in_query": [r"Verfahren", r"procedure", r"Prozess", r"Klage", r"Gericht"],
        "inject": [
            "Art. 176 Abs. 1 ZPO",    # witness protocol
            "Art. 181 Abs. 3 ZPO",    # physical evidence
            "Art. 300 ZPO",           # child representation
            "Art. 405 Abs. 1 ZPO",    # transitional law for appeals
        ],
    },
    "debt_enforcement": {
        "detect_in_predictions": ["SchKG"],
        "detect_in_query": [r"Betreibung", r"enforcement", r"SchKG", r"Konkurs", r"bankruptcy",
                            r"Schuld"],
        "inject": [
            "Art. 67 Abs. 1 SchKG",   # enforcement request
        ],
    },
    "criminal_general": {
        "detect_in_predictions": ["StGB"],
        "detect_in_query": [r"Straftat", r"offense", r"Vergehen", r"Verbrechen"],
        "inject": [
            "Art. 292 StGB",          # disobedience to official orders
        ],
    },
    "matrimonial_property": {
        "detect_in_predictions": ["ZGB"],
        "detect_in_query": [r"Güter", r"Errungenschaft", r"matrimonial property", r"Ehegatt"],
        "inject": [
            "Art. 197 Abs. 1 ZGB",    # definition of acquisitions
        ],
    },
    "signature": {
        "detect_in_predictions": ["OR"],
        "detect_in_query": [r"Unterschrift", r"sign", r"Urkunde", r"document"],
        "inject": [
            "Art. 15 OR",             # signature alternatives
        ],
    },
}


def load_predictions(path: Path) -> dict[str, set[str]]:
    with open(path) as f:
        reader = csv.DictReader(f)
        key = "predicted_citations" if "predicted_citations" in reader.fieldnames else "citations"
        return {row["query_id"]: set(c.strip() for c in row[key].split(";") if c.strip()) for row in reader}


def get_statute(cite: str) -> str | None:
    if not cite.startswith("Art."):
        return None
    parts = cite.split()
    return parts[-1] if len(parts) > 1 else None


def detect_proceeding_types(query: str, predictions: set[str]) -> list[str]:
    """Detect which proceeding types apply based on query text and predictions."""
    pred_statutes = {get_statute(c) for c in predictions if get_statute(c)}
    matched = []

    for proc_type, rules in PROCEEDING_RULES.items():
        # Check if any detection statute is in predictions
        statute_match = any(s in pred_statutes for s in rules.get("detect_in_predictions", []))

        # Check if any detection pattern is in query
        query_match = any(re.search(p, query, re.IGNORECASE) for p in rules.get("detect_in_query", []))

        if statute_match and query_match:
            matched.append(proc_type)

    return matched


def inject_procedural(
    predictions: dict[str, set[str]],
    queries: dict[str, str],
    max_inject_per_type: int = 5,
) -> dict[str, set[str]]:
    """Inject procedural citations based on detected proceeding types."""
    result = {}
    total_injected = 0

    for qid in predictions:
        preds = set(predictions[qid])
        query = queries.get(qid, "")

        proc_types = detect_proceeding_types(query, preds)
        injected = set()

        for proc_type in proc_types:
            rules = PROCEEDING_RULES[proc_type]
            for cite in rules["inject"][:max_inject_per_type]:
                if cite not in preds:
                    injected.add(cite)

        result[qid] = preds | injected
        if injected:
            total_injected += len(injected)
            print(f"  {qid}: +{len(injected)} procedural ({', '.join(sorted(proc_types))})")
            for c in sorted(injected):
                print(f"    + {c}")

    print(f"\nTotal injected: {total_injected}")
    return result


def write_csv(predictions: dict[str, set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            writer.writerow([qid, ";".join(sorted(predictions[qid]))])


def macro_f1(predictions: dict[str, set[str]], gold: dict[str, set[str]]) -> float:
    f1s = []
    for qid in gold:
        p = predictions.get(qid, set())
        g = gold[qid]
        if not p and not g: f1s.append(1.0); continue
        if not p or not g: f1s.append(0.0); continue
        tp = len(p & g)
        pr = tp / len(p)
        rc = tp / len(g)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
    return sum(f1s) / len(f1s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--gold", type=Path, default=BASE / "data/val.csv")
    parser.add_argument("--max-inject", type=int, default=5)
    args = parser.parse_args()

    predictions = load_predictions(args.baseline)

    # Load queries
    data_file = BASE / "data" / f"{args.split}.csv"
    queries = {}
    with open(data_file) as f:
        for row in csv.DictReader(f):
            queries[row["query_id"]] = row["query"]

    print(f"Loaded {len(predictions)} predictions, {len(queries)} queries")
    print()

    result = inject_procedural(predictions, queries, max_inject_per_type=args.max_inject)
    write_csv(result, args.output)
    print(f"\nWrote: {args.output}")

    # Evaluate if gold available
    if args.gold.exists() and args.split == "val":
        gold = {}
        with open(args.gold) as f:
            for row in csv.DictReader(f):
                gold[row["query_id"]] = set(row["gold_citations"].split(";"))

        baseline_f1 = macro_f1(predictions, gold)
        injected_f1 = macro_f1(result, gold)
        print(f"\nBaseline val F1: {baseline_f1:.6f}")
        print(f"Injected val F1: {injected_f1:.6f} ({(injected_f1 - baseline_f1) * 100:+.2f}pp)")


if __name__ == "__main__":
    main()
