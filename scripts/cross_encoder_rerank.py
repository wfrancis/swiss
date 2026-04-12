#!/usr/bin/env python3
"""
Cross-encoder re-ranker: score (query, citation_text) pairs and re-select.

This replaces the rule-based selector with a learned re-ranker that
actually READS the law/court text alongside the query.

Step 1: Pre-filter to top N candidates per query by final_score
Step 2: Score each (query, citation_text) pair with cross-encoder
Step 3: Select top-K by cross-encoder score
Step 4: Apply procedural injection

Zero API calls. ~1-5 minutes on CPU.
"""
import csv, json, pickle, re, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import CrossEncoder

BASE = Path(__file__).resolve().parent.parent


def load_preds(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        key = "predicted_citations" if "predicted_citations" in reader.fieldnames else "citations"
        return {row["query_id"]: set(c.strip() for c in row[key].split(";") if c.strip()) for row in reader}


def macro_f1(preds, gold):
    f1s = []
    for qid in gold:
        p = preds.get(qid, set()); g = gold[qid]
        if not p and not g: f1s.append(1.0); continue
        if not p or not g: f1s.append(0.0); continue
        tp = len(p & g); pr = tp / len(p); rc = tp / len(g)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
    return sum(f1s) / len(f1s)


def load_law_texts():
    texts = {}
    with open(BASE / "data/laws_de.csv") as f:
        for row in csv.DictReader(f):
            texts[row["citation"]] = row["text"][:500]
    return texts


def load_court_texts():
    texts = {}
    with open(BASE / "data/court_considerations.csv") as f:
        for row in csv.DictReader(f):
            texts[row["citation"]] = row["text"][:500]
    return texts


def load_queries(split):
    queries = {}
    with open(BASE / f"data/{split}.csv") as f:
        for row in csv.DictReader(f):
            queries[row["query_id"]] = row["query"]
    return queries


def load_bundles(path):
    data = json.load(open(path))
    bundles = {}
    for b in data["bundles"]:
        qid = b["query_id"]
        candidates = []
        for c in b["candidates"]:
            candidates.append({
                "citation": c["citation"],
                "kind": c.get("kind", "law"),
                "final_score": float(c.get("final_score", 0) or 0),
                "judge_label": c.get("judge_label", "reject"),
                "judge_confidence": float(c.get("judge_confidence", 0) or 0),
            })
        candidates.sort(key=lambda x: -x["final_score"])
        bundles[qid] = {
            "candidates": candidates,
            "estimated_count": b.get("estimated_count", 20),
        }
    return bundles


def rerank_with_cross_encoder(
    model, queries, bundles, law_texts, court_texts,
    top_n_candidates=100, batch_size=64,
):
    """Score top-N candidates per query with cross-encoder."""
    all_scores = {}

    for qid in sorted(bundles):
        query_text = queries.get(qid, "")[:1000]
        candidates = bundles[qid]["candidates"][:top_n_candidates]

        if not candidates:
            all_scores[qid] = []
            continue

        pairs = []
        for c in candidates:
            cite = c["citation"]
            text = law_texts.get(cite, court_texts.get(cite, ""))
            if text:
                pair_text = f"{cite}: {text[:400]}"
            else:
                pair_text = cite
            pairs.append((query_text, pair_text))

        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: -x[1])
        all_scores[qid] = scored_candidates

    return all_scores


def select_from_scores(scored, bundles, target_mult=1.0, bias=0, min_out=8, max_out=40, court_cap_frac=0.25):
    predictions = {}
    for qid, scored_cands in scored.items():
        est = bundles[qid]["estimated_count"]
        target = round(est * target_mult + bias)
        target = max(min_out, min(max_out, target))

        selected = set()
        courts = 0
        court_cap = max(0, round(target * court_cap_frac))

        for cand, score in scored_cands:
            if len(selected) >= target:
                break
            if cand["kind"] == "court" and courts >= court_cap:
                continue
            selected.add(cand["citation"])
            if cand["kind"] == "court":
                courts += 1

        predictions[qid] = selected
    return predictions


def apply_procedural(predictions, split):
    try:
        proc_cache = json.load(open(BASE / "precompute/llm_procedural_cache.json"))
        corpus = set()
        with open(BASE / "data/laws_de.csv") as f:
            for row in csv.DictReader(f):
                corpus.add(row["citation"])
        corpus_by_base = {}
        for cite in corpus:
            if cite.startswith("Art."):
                m = re.match(r"(Art\.\s+\d+[a-z]?)\b.*?(\S+)\s*$", cite)
                if m:
                    base = f"{m.group(1)} {m.group(2)}"
                    corpus_by_base.setdefault(base, []).append(cite)

        for qid in predictions:
            cls = proc_cache.get(f"{split}_{qid}", {})
            if cls.get("confidence", 0) < 0.7:
                continue
            for c in cls.get("citations", []):
                if "BGG" in c:
                    continue
                if c in corpus:
                    predictions[qid].add(c)
                else:
                    m2 = re.match(r"(Art\.\s+\d+[a-z]?)\b\s*(\S+)\s*$", c)
                    if m2:
                        bk = f"{m2.group(1)} {m2.group(2)}"
                        vs = corpus_by_base.get(bk, [])
                        if len(vs) == 1:
                            predictions[qid].add(vs[0])
                        else:
                            a1 = [v for v in vs if "Abs. 1" in v]
                            if a1:
                                predictions[qid].add(a1[0])
    except Exception as e:
        print(f"Procedural injection failed: {e}")
    return predictions


def write_csv(predictions, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "predicted_citations"])
        for qid in sorted(predictions):
            w.writerow([qid, ";".join(sorted(predictions[qid]))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--val-bundles", type=Path, default=BASE / "artifacts/v11/val_v11_strict_v1/judged_bundles.json")
    parser.add_argument("--test-bundles", type=Path, default=BASE / "artifacts/v11/test_v11_strict_v1/judged_bundles.json")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading cross-encoder: {args.model}", flush=True)
    model = CrossEncoder(args.model, max_length=512)

    print("Loading texts...", flush=True)
    law_texts = load_law_texts()
    court_texts = load_court_texts()
    print(f"  {len(law_texts)} laws, {len(court_texts)} courts", flush=True)

    gold = {}
    with open(BASE / "data/val.csv") as f:
        for row in csv.DictReader(f):
            gold[row["query_id"]] = set(row["gold_citations"].split(";"))

    # Val
    print("\n=== VAL ===", flush=True)
    val_queries = load_queries("val")
    val_bundles = load_bundles(args.val_bundles)

    t0 = time.time()
    val_scored = rerank_with_cross_encoder(model, val_queries, val_bundles, law_texts, court_texts, args.top_n, args.batch_size)
    print(f"  Scored in {time.time()-t0:.1f}s", flush=True)

    # Sweep selection params
    baseline_f1 = 0.338499
    best_f1 = 0
    best_params = None

    for tm in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for bias in [-4, -2, 0, 2, 4, 8]:
            for mino in [5, 8, 10, 15]:
                for maxo in [20, 25, 30, 35, 40, 50]:
                    if mino >= maxo:
                        continue
                    for ccf in [0.15, 0.25, 0.33, 0.50]:
                        preds = select_from_scores(val_scored, val_bundles, tm, bias, mino, maxo, ccf)
                        preds = apply_procedural(dict(preds), "val")
                        f1 = macro_f1(preds, gold)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = (tm, bias, mino, maxo, ccf)

    tm, bias, mino, maxo, ccf = best_params
    print(f"  Best val F1: {best_f1:.6f} ({(best_f1-baseline_f1)*100:+.2f}pp vs baseline)", flush=True)
    print(f"  Params: tm={tm}, bias={bias}, min={mino}, max={maxo}, court_cap={ccf}", flush=True)

    # Write val
    val_preds = select_from_scores(val_scored, val_bundles, tm, bias, mino, maxo, ccf)
    val_preds = apply_procedural(val_preds, "val")
    write_csv(val_preds, BASE / "submissions/val_pred_crossencoder.csv")
    avg = sum(len(p) for p in val_preds.values()) / len(val_preds)
    print(f"  Wrote val_pred_crossencoder.csv (avg {avg:.1f} cites)", flush=True)

    # Test
    print("\n=== TEST ===", flush=True)
    test_queries = load_queries("test")
    test_bundles = load_bundles(args.test_bundles)

    t0 = time.time()
    test_scored = rerank_with_cross_encoder(model, test_queries, test_bundles, law_texts, court_texts, args.top_n, args.batch_size)
    print(f"  Scored in {time.time()-t0:.1f}s", flush=True)

    test_preds = select_from_scores(test_scored, test_bundles, tm, bias, mino, maxo, ccf)
    test_preds = apply_procedural(test_preds, "test")
    write_csv(test_preds, BASE / "submissions/test_submission_crossencoder.csv")
    avg = sum(len(p) for p in test_preds.values()) / len(test_preds)
    print(f"  Wrote test_submission_crossencoder.csv (avg {avg:.1f} cites)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
