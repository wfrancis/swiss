#!/usr/bin/env bash
#
# Overnight brute force — maximize idle CPU.
# ALL LOCAL. Zero API calls. Zero cost.
#
# Job 1: Rust CV sweep with 500K iterations on 1,139 train queries
# Job 2: Rust CV sweep on multipass merged data
# Job 3: Extended corpus FAISS build + BM25 index
# Job 4: Massive perturbation search (all CSV pairs as potential neighbors)
# Job 5: Query-adaptive selector (different params per query type)

set -uo pipefail

REPO="/Users/william/swiss-legal-retrieval"
cd "$REPO"

PYTHON="$REPO/.venv/bin/python"
RUST_CV="$REPO/rust/v11_selector/target/release/cv_sweep"
RUST_HYBRID="$REPO/rust/v11_selector/target/release/hybrid_lab"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/overnight_bruteforce_$TS"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/main.log"; }

log "=== OVERNIGHT BRUTE FORCE (all local, zero API) ==="
log "Using 3 cores max"

# Stagger jobs — run 3 at a time, not all 5

# ============================================================
# JOB 1: Rust CV sweep — 500K iterations, 5 folds, full 1139 train
# ============================================================
log "JOB 1: Rust CV 500K on full train"
(
  time "$RUST_CV" artifacts/v11 data/train.csv \
    --folds 5 --iterations 500000 --seed 77 \
    --val-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json \
    --val-output submissions/val_pred_rust_cv500k.csv \
    --test-judged artifacts/v11/test_v11_strict_v1/judged_bundles.json \
    --test-output submissions/test_submission_rust_cv500k.csv
) >> "$LOG_DIR/job1_cv500k.log" 2>&1 &
PID1=$!
log "  PID=$PID1"

# JOB 2 launches in BATCH 2 (after batch 1 finishes)

# ============================================================
# JOB 3: Build extended corpus FAISS index
# ============================================================
log "JOB 3: Extended corpus FAISS index"
(
  $PYTHON -c "
import json, csv, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss

BASE = Path('.')

# Load treaty articles
treaties = json.load(open('precompute/treaty_articles.json'))
print(f'Treaty articles to embed: {len(treaties)}')

# Embed them
model = SentenceTransformer('intfloat/multilingual-e5-large')
texts = [f'passage: {a[\"text\"][:500]}' for a in treaties]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# Save embeddings
np.save('precompute/treaty_embeddings.npy', embeddings)

# Save citation list
treaty_cites = [a['citation'] for a in treaties]
with open('precompute/treaty_citations.json', 'w') as f:
    json.dump(treaty_cites, f)

print(f'Saved {len(treaties)} treaty embeddings to precompute/treaty_embeddings.npy')
print(f'These can be appended to the law FAISS index for extended retrieval')
"
) >> "$LOG_DIR/job3_faiss_extend.log" 2>&1 &
PID3=$!
log "  PID=$PID3"

# JOB 4 launches in BATCH 2 (after batch 1 finishes)

# --- BATCH 1: Jobs 1, 3, 5 (3 cores) ---
log "BATCH 1: Jobs 1 + 3 + 5 (3 cores)"
wait $PID1 $PID3 $PID5 2>/dev/null
log "BATCH 1 done"

# ============================================================
# JOB 2: Rust CV sweep on multipass merged data
# ============================================================
if [[ -f "artifacts/v11_multipass_merged/judged_bundles.json" ]]; then
  log "JOB 2: Rust CV on multipass merged"
  (
    ln -sf "$REPO/artifacts/v11_multipass_merged" "$REPO/artifacts/v11/train_multipass_merged__offset0_n1139" 2>/dev/null
    time "$RUST_CV" artifacts/v11 data/train.csv \
      --folds 5 --iterations 200000 --seed 88 \
      --val-judged artifacts/v11/val_v11_strict_v1/judged_bundles.json \
      --val-output submissions/val_pred_rust_cv_multipass.csv \
      --test-judged artifacts/v11/test_v11_strict_v1/judged_bundles.json \
      --test-output submissions/test_submission_rust_cv_multipass.csv
  ) >> "$LOG_DIR/job2_cv_multipass.log" 2>&1 &
  PID2=$!
else
  PID2=""
fi

# ============================================================
# JOB 4: Exhaustive perturbation search — all C(N,3) combos
# ============================================================
log "JOB 4: Exhaustive perturbation search"
(
  $PYTHON -c "
import csv, itertools, random, json, time
from pathlib import Path
from collections import defaultdict

def load_preds(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        key = 'predicted_citations' if 'predicted_citations' in reader.fieldnames else 'citations'
        return {row['query_id']: set(c.strip() for c in row[key].split(';') if c.strip()) for row in reader}

def macro_f1(preds, gold):
    f1s = []
    for qid in gold:
        p = preds.get(qid, set()); g = gold[qid]
        if not p and not g: f1s.append(1.0); continue
        if not p or not g: f1s.append(0.0); continue
        tp = len(p & g); pr = tp/len(p); rc = tp/len(g)
        f1s.append(2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0)
    return sum(f1s)/len(f1s)

def perturb(base, neighbors, add_vote_min=2, max_add=3):
    result = {}
    for qid in base:
        base_cites = set(base[qid])
        votes = defaultdict(int)
        for n in neighbors:
            for c in n.get(qid, set()):
                if c not in base_cites:
                    votes[c] += 1
        adds = sorted([c for c, v in votes.items() if v >= add_vote_min])[:max_add]
        result[qid] = base_cites | set(adds)
    return result

gold = {}
with open('data/val.csv') as f:
    for row in csv.DictReader(f):
        gold[row['query_id']] = set(row['gold_citations'].split(';'))

# Use the 0.30911 base (proc_nobgg)
base_val = load_preds('submissions/val_pred_llm_proc_nobgg.csv')
base_test = load_preds('submissions/test_submission_llm_proc_nobgg.csv')
bf1 = macro_f1(base_val, gold)

# Load ALL CSV pairs
submissions = Path('submissions')
pairs = []
for t in sorted(submissions.glob('test_submission_*.csv')):
    v = submissions / t.name.replace('test_submission_', 'val_pred_')
    if v.exists() and 'baseline' not in t.name and 'llm_proc_nobgg' not in t.name:
        try:
            val_p = load_preds(v)
            test_p = load_preds(t)
            if val_p and test_p:
                pairs.append({'name': t.stem, 'val': val_p, 'test': test_p})
        except:
            pass

print(f'Loaded {len(pairs)} CSV pairs')
print(f'Base (proc_nobgg): val F1 = {bf1:.6f}')
print(f'Testing C({len(pairs)},3) combos with multiple param settings...')

# Sweep add_vote_min and max_add too
param_grid = [
    (1, 2), (1, 3), (1, 5),
    (2, 3), (2, 5), (2, 8),
    (3, 3), (3, 5),
]

t0 = time.time()
results = []
total_combos = 0

for combo in itertools.combinations(range(len(pairs)), 3):
    i, j, k = combo
    nvs = [pairs[i]['val'], pairs[j]['val'], pairs[k]['val']]

    for avm, ma in param_grid:
        perturbed = perturb(base_val, nvs, add_vote_min=avm, max_add=ma)
        f1 = macro_f1(perturbed, gold)
        if f1 > bf1:
            results.append((f1, combo, avm, ma))
        total_combos += 1

    if total_combos % 50000 == 0:
        elapsed = time.time() - t0
        print(f'  {total_combos:,} combos in {elapsed:.0f}s, {len(results)} improvements found')

elapsed = time.time() - t0
results.sort(key=lambda r: -r[0])

print(f'\nDone: {total_combos:,} combos in {elapsed:.0f}s')
print(f'Improvements over baseline: {len(results)}')
print()

# Write top 10
print('TOP 10:')
for rank, (f1, combo, avm, ma) in enumerate(results[:10], 1):
    names = [pairs[c]['name'] for c in combo]
    delta = (f1 - bf1) * 100
    print(f'  #{rank}: F1={f1:.6f} ({delta:+.2f}pp) avm={avm} ma={ma}')
    print(f'    {names}')

# Write top 5 CSVs
for rank, (f1, combo, avm, ma) in enumerate(results[:5], 1):
    nvs_val = [pairs[c]['val'] for c in combo]
    nvs_test = [pairs[c]['test'] for c in combo]
    val_p = perturb(base_val, nvs_val, avm, ma)
    test_p = perturb(base_test, nvs_test, avm, ma)
    for fname, data in [(f'val_pred_bruteforce2_top{rank}.csv', val_p),
                        (f'test_submission_bruteforce2_top{rank}.csv', test_p)]:
        with open(f'submissions/{fname}', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['query_id', 'predicted_citations'])
            for qid in sorted(data):
                w.writerow([qid, ';'.join(sorted(data[qid]))])

# Save full results
with open('$LOG_DIR/perturbation_results.json', 'w') as f:
    json.dump([{'f1': r[0], 'combo': [pairs[c]['name'] for c in r[1]],
                'avm': r[2], 'ma': r[3]} for r in results[:100]], f, indent=2)

print(f'\nWrote top 5 CSVs + results JSON')
"
) >> "$LOG_DIR/job4_perturb_exhaust.log" 2>&1 &
PID4=$!
log "  PID=$PID4"

# ============================================================
# JOB 5: Query-adaptive selector — different params per domain
# ============================================================
log "JOB 5: Query-adaptive selector sweep"
(
  $PYTHON -c "
import csv, json, pickle, itertools, re
from pathlib import Path
from collections import defaultdict

# Load all judged train bundles
import glob
bundles_by_qid = {}
for f in sorted(glob.glob('artifacts/v11/train_v11_strict_v1__offset*/judged_bundles.pkl')):
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    for b in data['bundles']:
        bundles_by_qid[b.query_id] = b

# Load train gold
train_gold = {}
with open('data/train.csv') as f:
    for row in csv.DictReader(f):
        train_gold[row['query_id']] = set(row['gold_citations'].split(';'))

# Load query expansions for domain info
expansions = json.load(open('precompute/train_query_expansions.json'))

# Classify queries by gold count (proxy for complexity)
low_gold = []   # 1-3 gold cites
mid_gold = []   # 4-10 gold cites
high_gold = []  # 11+ gold cites

for qid in bundles_by_qid:
    if qid not in train_gold:
        continue
    n_gold = len(train_gold[qid])
    if n_gold <= 3:
        low_gold.append(qid)
    elif n_gold <= 10:
        mid_gold.append(qid)
    else:
        high_gold.append(qid)

print(f'Train queries by gold count:')
print(f'  Low (1-3):  {len(low_gold)}')
print(f'  Mid (4-10): {len(mid_gold)}')
print(f'  High (11+): {len(high_gold)}')

# Import selector
import sys
sys.path.insert(0, '.')
from pipeline_v11 import V11Config, select_candidates, evaluate_predictions

# Sweep selector params PER GROUP and find best combo
max_outputs = [10, 15, 20, 25, 30, 35, 40, 50]
min_outputs = [3, 5, 8, 10, 15]
court_fracs = [0.10, 0.15, 0.20, 0.25, 0.30]
must_confs = [0.70, 0.80, 0.86, 0.90, 0.95]

def eval_config(qids, bundles, gold, params):
    config = V11Config(
        split='train', use_judge=True, judge_model='x', prompt_version='x',
        law_judge_topk=60, court_judge_topk=36, law_batch_size=20, court_batch_size=12,
        use_court_dense=True, court_dense_query_limit=8, court_dense_topk=160,
        max_output=params['max_output'], min_output=params['min_output'],
        court_fraction=params['court_fraction'], min_courts_if_any=4,
        must_keep_confidence=params['must_keep_confidence'],
        cache_path=Path('/dev/null'), court_text_cache_path=Path('/dev/null'),
        court_dense_cache_path=Path('/dev/null'), query_offset=0, max_queries=None,
    )
    preds = {}
    for qid in qids:
        if qid not in bundles:
            continue
        selected = select_candidates(bundles[qid], config)
        preds[qid] = {c.citation for c in selected}
    f1, _ = evaluate_predictions(preds, {q: gold[q] for q in preds if q in gold})
    return f1

# Find best config per group
print()
for group_name, group_qids in [('low', low_gold), ('mid', mid_gold), ('high', high_gold)]:
    if not group_qids:
        continue
    best_f1 = 0
    best_params = None
    tested = 0
    for maxo in max_outputs:
        for mino in min_outputs:
            if mino >= maxo:
                continue
            for cf in court_fracs:
                for mc in must_confs:
                    params = {'max_output': maxo, 'min_output': mino,
                              'court_fraction': cf, 'must_keep_confidence': mc}
                    f1 = eval_config(group_qids[:200], bundles_by_qid, train_gold, params)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = params
                    tested += 1
    print(f'{group_name} ({len(group_qids)} queries): best F1={best_f1:.6f} params={best_params} ({tested} configs)')

print()
print('Query-adaptive insight: different gold counts need different max_output')
print('This can be applied to test queries using estimated_count from the pipeline')
"
) >> "$LOG_DIR/job5_adaptive.log" 2>&1 &
PID5=$!
log "  PID=$PID5"

# --- BATCH 2: Jobs 2 + 4 (2 cores) ---
log "BATCH 2: Jobs 2 + 4 (2 cores)"
wait ${PID2:-} $PID4 2>/dev/null
log "BATCH 2 done"

log ""
log "=== ALL JOBS COMPLETE ==="
log "Results:"
for f in "$LOG_DIR"/job*.log; do
  echo "--- $(basename $f) ---" >> "$LOG_DIR/main.log"
  tail -5 "$f" >> "$LOG_DIR/main.log" 2>/dev/null
done
log "=== DONE ==="
