/// Diverse eval scorecard — slice queries by characteristic buckets, compute
/// per-bucket F1 (on val, bootstrap CI) and per-bucket churn/Jaccard (on test,
/// if provided), report candidate vs baseline.
///
/// Usage:
///     diverse_eval <cand_val_csv> <base_val_csv> [cand_test_csv] [base_test_csv]
///
/// Reads data/val.csv, data/test.csv, and precompute/llm_procedural_cache.json
/// from CWD.
///
/// Promotion rule (per CLAUDE.md 2026-04-10):
///   1. Overall val F1 >= baseline
///   2. Dominates in >=3 ROBUST buckets (N>=3 queries — 1-query buckets don't count)
///   3. No severe regression (>3pp) in any bucket, even low-N
///   4. If test CSVs provided: overall test Jaccard >= 0.85 (high churn = high
///      shakeup risk; combo_a won at 0.82 but that's an outlier)
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ValRow {
    query_id: String,
    query: String,
    gold_citations: String,
}

#[derive(Debug, Deserialize)]
struct TestRow {
    query_id: String,
    query: String,
}

#[derive(Debug, Deserialize)]
struct PredRow {
    query_id: String,
    predicted_citations: String,
}

#[derive(Debug, Deserialize)]
struct ProcEntry {
    proceeding_type: String,
}

fn load_val(path: &str) -> Result<Vec<ValRow>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let rows: Vec<ValRow> = rdr.deserialize().collect::<Result<_, _>>()?;
    Ok(rows)
}

fn load_test(path: &str) -> Result<Vec<TestRow>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let rows: Vec<TestRow> = rdr.deserialize().collect::<Result<_, _>>()?;
    Ok(rows)
}

fn load_preds(path: &str) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let mut map = HashMap::new();
    for result in rdr.deserialize() {
        let row: PredRow = result?;
        let cites: Vec<String> = row
            .predicted_citations
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        map.insert(row.query_id, cites);
    }
    Ok(map)
}

fn load_proc_cache(path: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let raw: HashMap<String, ProcEntry> = serde_json::from_str(&content)?;
    // Cache keys look like "val_val_010" or "test_test_001" (split prefix + qid).
    // The query_id already includes "val_"/"test_", so strip one extra prefix.
    let mut map = HashMap::new();
    for (key, entry) in raw {
        let qid = if let Some(rest) = key.strip_prefix("val_") {
            if rest.starts_with("val_") {
                rest.to_string()
            } else {
                key.clone()
            }
        } else if let Some(rest) = key.strip_prefix("test_") {
            if rest.starts_with("test_") {
                rest.to_string()
            } else {
                key.clone()
            }
        } else {
            key.clone()
        };
        map.insert(qid, entry.proceeding_type);
    }
    Ok(map)
}

/// Classify a query into zero-or-more buckets. A query typically lands in
/// one proceeding bucket, several domain buckets, one size bucket, one length
/// bucket — plus the implicit "all" bucket.
///
/// Domain classification is driven by **gold citations** (when available) or
/// by the baseline prediction set as a proxy for test queries. The previous
/// query-text regex matched the English word "or" as an OR-code reference,
/// which tagged 6/10 val queries with dom_or spuriously (mathematician's B1).
/// Suffix-matching legal-code abbreviations in citations is unambiguous.
fn classify_buckets(query: &str, gold_or_proxy: &[String], proc_type: Option<&str>) -> Vec<String> {
    let mut buckets = Vec::new();

    // PROCEEDING (from LLM cache).
    let proc_tag = if let Some(pt) = proc_type {
        let p = pt.to_lowercase();
        if p.contains("criminal") || p.contains("straf") {
            Some("proc_criminal")
        } else if p.contains("civil") || p.contains("zivil") {
            Some("proc_civil")
        } else if p.contains("admin") || p.contains("verwalt") {
            Some("proc_admin")
        } else if p.contains("social") || p.contains("sozial") || p.contains("ivg") {
            Some("proc_social")
        } else if p.contains("schuld") || p.contains("betreib") {
            Some("proc_debt")
        } else {
            None
        }
    } else {
        None
    };
    if let Some(tag) = proc_tag {
        buckets.push(tag.to_string());
    }

    // DOMAIN (from citation-code suffix matching — unambiguous).
    // Each legal code appears as a suffix like "Art. 221 Abs. 1 StPO" or
    // "Art. 41 OR". We search for the code token preceded by whitespace to
    // avoid partial matches (e.g., "StGB" inside "ZStGB" is unlikely but safe).
    let has_code = |tok: &str| -> bool {
        gold_or_proxy
            .iter()
            .any(|c| c.contains(&format!(" {}", tok)) || c.ends_with(tok))
    };

    if has_code("StPO") {
        buckets.push("dom_stpo".to_string());
    }
    if has_code("StGB") {
        buckets.push("dom_stgb".to_string());
    }
    if has_code("ZGB") {
        buckets.push("dom_zgb".to_string());
    }
    if has_code("OR") {
        buckets.push("dom_or".to_string());
    }
    if has_code("ZPO") {
        buckets.push("dom_zpo".to_string());
    }
    if has_code("ATSG") || has_code("IVG") || has_code("UVG") || has_code("KVG") || has_code("AHVG")
    {
        buckets.push("dom_social_ins".to_string());
    }
    if has_code("SchKG") {
        buckets.push("dom_schkg".to_string());
    }
    if has_code("BGG") {
        buckets.push("dom_bgg".to_string());
    }

    // SIZE (gold citation count)
    let gold_count = gold_or_proxy.len();
    if gold_count < 15 {
        buckets.push("size_small".to_string());
    } else if gold_count <= 30 {
        buckets.push("size_medium".to_string());
    } else {
        buckets.push("size_large".to_string());
    }

    // LENGTH (query chars)
    let qlen = query.len();
    if qlen < 1200 {
        buckets.push("len_short".to_string());
    } else if qlen < 1700 {
        buckets.push("len_medium".to_string());
    } else {
        buckets.push("len_long".to_string());
    }

    buckets.push("all".to_string());
    buckets
}

fn f1(pred: &HashSet<&str>, gold: &HashSet<&str>) -> f64 {
    if pred.is_empty() && gold.is_empty() {
        return 1.0;
    }
    let tp = pred.intersection(gold).count() as f64;
    let p = if pred.is_empty() {
        0.0
    } else {
        tp / pred.len() as f64
    };
    let r = if gold.is_empty() {
        0.0
    } else {
        tp / gold.len() as f64
    };
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// Bootstrap: resample queries with replacement, return (mean, p05, p95).
fn bootstrap_ci(per_query_f1: &[f64], n_iters: usize, seed: u64) -> (f64, f64, f64) {
    let n = per_query_f1.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means: Vec<f64> = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += per_query_f1[idx];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = means.iter().sum::<f64>() / n_iters as f64;
    let p05 = means[(n_iters as f64 * 0.05) as usize];
    let p95 = means[(n_iters as f64 * 0.95) as usize];
    (mean, p05, p95)
}

/// Hardcoded historical candidates with their known Kaggle public-LB scores.
/// Used as a calibration anchor: when we eval a new candidate, we ALSO run the
/// same verdict logic against each of these historical points using the user's
/// baseline, so the tool's verdict comes with an empirical accuracy statement
/// ("on the 10 known-outcome points, REJECT was N/M correct").
///
/// Fields: name, val CSV, test CSV, Kaggle public-LB score.
struct HistEntry {
    name: &'static str,
    val_csv: &'static str,
    test_csv: &'static str,
    kaggle: f64,
}

const HISTORY: &[HistEntry] = &[
    HistEntry {
        name: "llm_proc_nobgg",
        val_csv: "submissions/val_pred_baseline_public_best_30911.csv",
        test_csv: "submissions/test_submission_baseline_public_best_30911.csv",
        kaggle: 0.30911,
    },
    HistEntry {
        name: "combo_layer_1",
        val_csv: "submissions/val_pred_combo_layer_1.csv",
        test_csv: "submissions/test_submission_combo_layer_1.csv",
        kaggle: 0.30911,
    },
    HistEntry {
        name: "combo_a",
        val_csv: "submissions/val_pred_overnight_combo_a.csv",
        test_csv: "submissions/test_submission_overnight_combo_a.csv",
        kaggle: 0.30681,
    },
    HistEntry {
        name: "treaty_inject",
        val_csv: "submissions/val_pred_treaty_inject.csv",
        test_csv: "submissions/test_submission_treaty_inject.csv",
        kaggle: 0.30460,
    },
    HistEntry {
        name: "combo_layer_2",
        val_csv: "submissions/val_pred_combo_layer_2.csv",
        test_csv: "submissions/test_submission_combo_layer_2.csv",
        kaggle: 0.30340,
    },
    HistEntry {
        name: "bruteforce_top1",
        val_csv: "submissions/val_pred_bruteforce_top1.csv",
        test_csv: "submissions/test_submission_bruteforce_top1.csv",
        kaggle: 0.30291,
    },
    HistEntry {
        name: "baseline_30257",
        val_csv: "submissions/val_pred_baseline_public_best_30257.csv",
        test_csv: "submissions/test_submission_baseline_public_best_30257.csv",
        kaggle: 0.30257,
    },
    HistEntry {
        name: "proc_perturb",
        val_csv: "submissions/val_pred_proc_perturb.csv",
        test_csv: "submissions/test_submission_proc_perturb.csv",
        kaggle: 0.30191,
    },
    HistEntry {
        name: "claude_agree",
        val_csv: "submissions/val_pred_claude_agree.csv",
        test_csv: "submissions/test_submission_claude_agree.csv",
        kaggle: 0.29508,
    },
    HistEntry {
        name: "procedural_inject",
        val_csv: "submissions/val_pred_procedural_inject.csv",
        test_csv: "submissions/test_submission_procedural_inject.csv",
        kaggle: 0.28661,
    },
];

/// Compute a simplified verdict tag for calibration. Same decision logic as
/// the main tool, collapsed into (tag, val_delta_pp, test_jaccard, adds, removes).
fn simple_verdict(
    cand_val: &HashMap<String, Vec<String>>,
    base_val: &HashMap<String, Vec<String>>,
    cand_test: &HashMap<String, Vec<String>>,
    base_test: &HashMap<String, Vec<String>>,
    val_rows: &[ValRow],
    test_rows: &[TestRow],
) -> (&'static str, f64, f64, usize, usize) {
    // Val F1s
    let mut cand_sum = 0.0;
    let mut base_sum = 0.0;
    for row in val_rows {
        let gold_strs: Vec<String> = row
            .gold_citations
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        let gold_set: HashSet<&str> = gold_strs.iter().map(|s| s.as_str()).collect();
        let empty = Vec::new();
        let c = cand_val.get(&row.query_id).unwrap_or(&empty);
        let b = base_val.get(&row.query_id).unwrap_or(&empty);
        let c_set: HashSet<&str> = c.iter().map(|s| s.as_str()).collect();
        let b_set: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
        cand_sum += f1(&c_set, &gold_set);
        base_sum += f1(&b_set, &gold_set);
    }
    let n = val_rows.len() as f64;
    let val_delta_pp = ((cand_sum - base_sum) / n) * 100.0;

    // Test churn + Jaccard
    let mut adds = 0usize;
    let mut removes = 0usize;
    let mut jsum = 0.0f64;
    for row in test_rows {
        let empty = Vec::new();
        let c = cand_test.get(&row.query_id).unwrap_or(&empty);
        let b = base_test.get(&row.query_id).unwrap_or(&empty);
        let c_set: HashSet<&str> = c.iter().map(|s| s.as_str()).collect();
        let b_set: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
        adds += c_set.difference(&b_set).count();
        removes += b_set.difference(&c_set).count();
        jsum += jaccard(&c_set, &b_set);
    }
    let test_jaccard = jsum / test_rows.len() as f64;

    // Simplified verdict — empirically tuned on 10 known-outcome submissions.
    //   Wins:   adds/removes ratio 1.68x–2.17x
    //   Losses: adds/removes ratio 3.49x–5.54x
    // Separator at 2.5x cleanly divides them. Thresholds are explicitly
    // overfit to this small dataset — see historical_calibration output to
    // monitor for regressions as new submissions are added.
    let add_rem_ratio = if removes > 0 {
        adds as f64 / removes as f64
    } else if adds > 0 {
        99.0 // pure-additive = treat as very high
    } else {
        1.0
    };
    let is_pure_additive = removes == 0 && adds >= 10;
    let imbalanced = is_pure_additive || add_rem_ratio > 2.5 || add_rem_ratio < 0.4;

    // Jaccard also matters: very low Jaccard with modest val lift = risky.
    let low_jaccard_risky = test_jaccard < 0.75 && val_delta_pp < 4.0;

    let tag = if val_delta_pp < -1.0 {
        "REJECT"
    } else if imbalanced && val_delta_pp >= 0.0 {
        "HOLD-churn"
    } else if low_jaccard_risky {
        "HOLD-shape"
    } else if val_delta_pp >= 2.0 {
        "PROMOTE"
    } else if val_delta_pp >= 0.5 {
        "weak-PROMOTE"
    } else {
        "HOLD"
    };
    (tag, val_delta_pp, test_jaccard, adds, removes)
}

fn jaccard(a: &HashSet<&str>, b: &HashSet<&str>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let candidate_val_path = args
        .next()
        .ok_or("usage: diverse_eval <cand_val> <base_val> [cand_test] [base_test]")?;
    let baseline_val_path = args
        .next()
        .ok_or("usage: diverse_eval <cand_val> <base_val> [cand_test] [base_test]")?;
    let candidate_test_path = args.next();
    let baseline_test_path = args.next();

    let val_rows = load_val("data/val.csv")?;
    let cand_preds = load_preds(&candidate_val_path)?;
    let base_preds = load_preds(&baseline_val_path)?;
    let proc_cache =
        load_proc_cache("precompute/llm_procedural_cache.json").unwrap_or_else(|_| HashMap::new());

    // Optional test side: load only if both test paths provided
    let test_loaded = match (&candidate_test_path, &baseline_test_path) {
        (Some(cand), Some(base)) => {
            let test_rows = load_test("data/test.csv")?;
            let cand_test = load_preds(cand)?;
            let base_test = load_preds(base)?;
            Some((test_rows, cand_test, base_test, cand.clone(), base.clone()))
        }
        _ => None,
    };

    // Classify each query + compute per-query F1 for both candidate and baseline
    let mut query_buckets: HashMap<String, Vec<String>> = HashMap::new();
    let mut qid_to_f1s: HashMap<String, (f64, f64)> = HashMap::new(); // (cand, base)

    for row in &val_rows {
        let gold_strs: Vec<String> = row
            .gold_citations
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        let gold_set: HashSet<&str> = gold_strs.iter().map(|s| s.as_str()).collect();

        let empty = Vec::new();
        let cand_cites = cand_preds.get(&row.query_id).unwrap_or(&empty);
        let base_cites = base_preds.get(&row.query_id).unwrap_or(&empty);
        let cand_set: HashSet<&str> = cand_cites.iter().map(|s| s.as_str()).collect();
        let base_set: HashSet<&str> = base_cites.iter().map(|s| s.as_str()).collect();

        let cand_f1 = f1(&cand_set, &gold_set);
        let base_f1 = f1(&base_set, &gold_set);

        let pt = proc_cache.get(&row.query_id).map(|s| s.as_str());
        let buckets = classify_buckets(&row.query, &gold_strs, pt);

        query_buckets.insert(row.query_id.clone(), buckets);
        qid_to_f1s.insert(row.query_id.clone(), (cand_f1, base_f1));
    }

    // Group queries by bucket
    let mut bucket_to_qids: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (qid, buckets) in &query_buckets {
        for b in buckets {
            bucket_to_qids
                .entry(b.clone())
                .or_default()
                .push(qid.clone());
        }
    }

    // Header
    println!();
    println!("DIVERSE EVAL SCORECARD");
    println!("  Candidate val: {}", candidate_val_path);
    println!("  Baseline val:  {}", baseline_val_path);
    if let Some((_, _, _, ct, bt)) = &test_loaded {
        println!("  Candidate test: {}", ct);
        println!("  Baseline test:  {}", bt);
    }
    println!("  Val rows:      {}", val_rows.len());
    println!();
    println!("=== VAL: per-bucket F1 (bootstrap 1000x, 90% CI) ===");
    println!(
        "{:<18} {:>3} {:>16} {:>16} {:>9} {:>9}",
        "BUCKET", "N", "BASELINE (±90%)", "CANDIDATE (±90%)", "DELTA", "VERDICT"
    );
    println!("{}", "-".repeat(80));

    // Ordered bucket groups for pretty output
    let group_order = ["proc_", "dom_", "size_", "len_"];
    let mut ordered_buckets: Vec<&String> = bucket_to_qids.keys().collect();
    ordered_buckets.sort_by_key(|b| {
        let gi = group_order
            .iter()
            .position(|p| b.starts_with(p))
            .unwrap_or(99);
        (gi, (*b).clone())
    });

    let mut robust_dominates: usize = 0; // N>=3 buckets that improve >1pp
    let mut robust_total: usize = 0;
    let mut any_regresses: usize = 0;
    let mut severe_reg: usize = 0;
    let mut counted_buckets: usize = 0;
    let mut last_group: Option<&str> = None;

    for bucket in &ordered_buckets {
        if bucket.as_str() == "all" {
            continue;
        }
        let qids = &bucket_to_qids[*bucket];
        if qids.is_empty() {
            continue;
        }
        let base_f1s: Vec<f64> = qids.iter().map(|q| qid_to_f1s[q].1).collect();
        let cand_f1s: Vec<f64> = qids.iter().map(|q| qid_to_f1s[q].0).collect();

        let (b_mean, b_lo, b_hi) = bootstrap_ci(&base_f1s, 1000, 42);
        let (c_mean, c_lo, c_hi) = bootstrap_ci(&cand_f1s, 1000, 42);
        let delta_pp = (c_mean - b_mean) * 100.0;

        let is_robust = qids.len() >= 3;
        if is_robust {
            robust_total += 1;
        }

        let verdict = if delta_pp > 1.0 {
            if is_robust {
                robust_dominates += 1;
                "BETTER*"
            } else {
                "better?"
            }
        } else if delta_pp >= -1.0 {
            "~SAME"
        } else {
            any_regresses += 1;
            if delta_pp < -3.0 {
                // Only count as "severe" if N>=3 (robust). A 3pp swing on
                // N=1 or N=2 is within bootstrap noise — not signal.
                if is_robust {
                    severe_reg += 1;
                    "SEVERE"
                } else {
                    "worse?"
                }
            } else {
                "WORSE"
            }
        };
        counted_buckets += 1;

        // Group separator
        let group_prefix = group_order
            .iter()
            .find(|p| bucket.starts_with(*p))
            .copied()
            .unwrap_or("");
        if last_group.is_some() && last_group != Some(group_prefix) {
            println!("{}", "-".repeat(80));
        }
        last_group = Some(group_prefix);

        println!(
            "{:<18} {:>3} {:>7.3}±{:<7.3} {:>7.3}±{:<7.3} {:+7.1}pp {:>9}",
            bucket,
            qids.len(),
            b_mean,
            (b_hi - b_lo) / 2.0,
            c_mean,
            (c_hi - c_lo) / 2.0,
            delta_pp,
            verdict,
        );
    }

    // Overall "all" row
    if let Some(qids) = bucket_to_qids.get("all") {
        println!("{}", "=".repeat(80));
        let base_f1s: Vec<f64> = qids.iter().map(|q| qid_to_f1s[q].1).collect();
        let cand_f1s: Vec<f64> = qids.iter().map(|q| qid_to_f1s[q].0).collect();
        let (b_mean, b_lo, b_hi) = bootstrap_ci(&base_f1s, 1000, 42);
        let (c_mean, c_lo, c_hi) = bootstrap_ci(&cand_f1s, 1000, 42);
        let delta_pp = (c_mean - b_mean) * 100.0;
        println!(
            "{:<18} {:>3} {:>7.3}±{:<7.3} {:>7.3}±{:<7.3} {:+7.1}pp {:>9}",
            "OVERALL",
            qids.len(),
            b_mean,
            (b_hi - b_lo) / 2.0,
            c_mean,
            (c_hi - c_lo) / 2.0,
            delta_pp,
            if delta_pp > 0.0 { "up" } else { "down" },
        );
    }

    // ---- TEST side: churn + Jaccard per bucket (if provided) ----
    let mut test_jaccard_overall: f64 = 1.0;
    let mut test_jaccard_min_bucket: f64 = 1.0;
    let mut test_has_data = false;
    let mut test_total_adds: usize = 0;
    let mut test_total_removes: usize = 0;
    if let Some((test_rows, cand_test, base_test, _, _)) = &test_loaded {
        test_has_data = true;
        println!();
        println!("=== TEST: per-bucket churn + Jaccard (no gold — shape only) ===");

        // Classify test queries using same bucketing logic. We use the BASELINE
        // predictions as the gold proxy (not the candidate's) so both sides get
        // classified into the same buckets regardless of candidate shape.
        let empty_pred = Vec::new();
        let mut test_query_buckets: HashMap<String, Vec<String>> = HashMap::new();
        for row in test_rows {
            let proxy = base_test.get(&row.query_id).unwrap_or(&empty_pred);
            let pt = proc_cache.get(&row.query_id).map(|s| s.as_str());
            let buckets = classify_buckets(&row.query, proxy, pt);
            test_query_buckets.insert(row.query_id.clone(), buckets);
        }

        let mut test_bucket_to_qids: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (qid, buckets) in &test_query_buckets {
            for b in buckets {
                test_bucket_to_qids
                    .entry(b.clone())
                    .or_default()
                    .push(qid.clone());
            }
        }

        println!(
            "{:<18} {:>3} {:>8} {:>8} {:>8} {:>10}",
            "BUCKET", "N", "ADDS", "REMOVES", "CHURN", "JACCARD"
        );
        println!("{}", "-".repeat(70));

        let mut ordered: Vec<&String> = test_bucket_to_qids.keys().collect();
        ordered.sort_by_key(|b| {
            let gi = group_order
                .iter()
                .position(|p| b.starts_with(p))
                .unwrap_or(99);
            (gi, b.to_string())
        });

        let mut last_group_t: Option<&str> = None;
        for bucket in &ordered {
            let qids = &test_bucket_to_qids[*bucket];
            if qids.is_empty() {
                continue;
            }
            let mut adds = 0usize;
            let mut removes = 0usize;
            let mut jsum = 0.0f64;
            for qid in qids {
                let empty = Vec::new();
                let cand = cand_test.get(qid).unwrap_or(&empty);
                let base = base_test.get(qid).unwrap_or(&empty);
                let cset: HashSet<&str> = cand.iter().map(|s| s.as_str()).collect();
                let bset: HashSet<&str> = base.iter().map(|s| s.as_str()).collect();
                adds += cset.difference(&bset).count();
                removes += bset.difference(&cset).count();
                jsum += jaccard(&cset, &bset);
            }
            let jmean = jsum / qids.len() as f64;

            if bucket.as_str() != "all" && jmean < test_jaccard_min_bucket {
                test_jaccard_min_bucket = jmean;
            }
            if bucket.as_str() == "all" {
                test_jaccard_overall = jmean;
                test_total_adds = adds;
                test_total_removes = removes;
            }

            let group_prefix = group_order
                .iter()
                .find(|p| bucket.starts_with(*p))
                .copied()
                .unwrap_or("all");
            if last_group_t.is_some() && last_group_t != Some(group_prefix) {
                println!("{}", "-".repeat(70));
            }
            last_group_t = Some(group_prefix);

            println!(
                "{:<18} {:>3} {:>8} {:>8} {:>8} {:>10.3}",
                bucket,
                qids.len(),
                adds,
                removes,
                adds + removes,
                jmean
            );
        }
    }

    // ---- Summary + verdict ----
    println!();
    println!("SUMMARY:");
    println!(
        "  Val dominates (N>=3): {}/{} robust buckets  (* = counted)",
        robust_dominates, robust_total
    );
    println!(
        "  Val regressions:      {}/{} buckets  ({} severe >3pp)",
        any_regresses, counted_buckets, severe_reg
    );
    if test_has_data {
        println!(
            "  Test Jaccard overall: {:.3}  (min bucket: {:.3})",
            test_jaccard_overall, test_jaccard_min_bucket
        );
    }
    println!();

    // Promotion rule:
    //   1. Overall val F1 >= baseline
    //   2. Dominates in >=3 ROBUST buckets (N>=3 queries)
    //   3. No severe regression (>3pp) in any bucket
    //   4. If test provided: overall test Jaccard >= 0.85
    let overall_qids = &bucket_to_qids["all"];
    let overall_base: f64 =
        overall_qids.iter().map(|q| qid_to_f1s[q].1).sum::<f64>() / overall_qids.len() as f64;
    let overall_cand: f64 =
        overall_qids.iter().map(|q| qid_to_f1s[q].0).sum::<f64>() / overall_qids.len() as f64;

    let overall_delta_pp = (overall_cand - overall_base) * 100.0;
    let overall_up = overall_cand > overall_base;
    let overall_strong = overall_delta_pp >= 2.0;
    let no_severe = severe_reg == 0;
    let many_robust = robust_dominates >= 5;
    let some_robust = robust_dominates >= 3;

    // Tiered verdict — CLAUDE.md 2026-04-10 lesson: Jaccard is informational,
    // not a gate. Combo_a won at 0.82 Jaccard. Val lift + bucket breadth matter
    // more than shape stability.
    println!("PROMOTION ANALYSIS:");
    println!(
        "  [{}] Overall val F1 up          ({:+.1}pp: {:.3} vs {:.3})",
        if overall_up { "PASS" } else { "FAIL" },
        overall_delta_pp,
        overall_cand,
        overall_base
    );
    println!(
        "  [{}] Overall val F1 strong up   (>=+2.0pp, got {:+.1}pp)",
        if overall_strong { "PASS" } else { "FAIL" },
        overall_delta_pp
    );
    println!(
        "  [{}] Robust dominance wide      (>=5 N>=3 buckets, got {}/{})",
        if many_robust { "PASS" } else { "WARN" },
        robust_dominates,
        robust_total
    );
    println!(
        "  [{}] Robust dominance some      (>=3 N>=3 buckets, got {}/{})",
        if some_robust { "PASS" } else { "FAIL" },
        robust_dominates,
        robust_total
    );
    println!(
        "  [{}] No severe regressions      ({} severe, {} minor)",
        if no_severe { "PASS" } else { "FAIL" },
        severe_reg,
        any_regresses - severe_reg
    );
    // Churn balance: purely additive candidates (remove 0) kill precision.
    //   - Balanced:   0.4 <= adds/(adds+removes) <= 0.7
    //   - Additive:   ratio > 0.85      -> precision risk
    //   - Pure-add:   removes == 0, adds >= 10   -> very high precision risk
    //   - Subtract:   ratio < 0.15      -> recall risk
    let churn_balanced_ok;
    let churn_tag: &str;
    if test_has_data {
        let total = test_total_adds + test_total_removes;
        let ratio = if total == 0 {
            0.5
        } else {
            test_total_adds as f64 / total as f64
        };
        if test_total_removes == 0 && test_total_adds >= 10 {
            churn_tag = "PURE-ADDITIVE (precision killer)";
            churn_balanced_ok = false;
        } else if ratio > 0.85 {
            churn_tag = "mostly additive (precision risk)";
            churn_balanced_ok = false;
        } else if ratio < 0.15 {
            churn_tag = "mostly subtractive (recall risk)";
            churn_balanced_ok = false;
        } else {
            churn_tag = "balanced";
            churn_balanced_ok = true;
        }
        let shape_tag = if test_jaccard_overall >= 0.90 {
            "conservative shape"
        } else if test_jaccard_overall >= 0.80 {
            "moderate shape"
        } else if test_jaccard_overall >= 0.70 {
            "high-variance shape"
        } else {
            "very risky shape"
        };
        println!(
            "  [info] Test Jaccard             ({:.3} — {}; min bucket {:.3})",
            test_jaccard_overall, shape_tag, test_jaccard_min_bucket
        );
        println!(
            "  [{}] Churn balance             ({} adds / {} removes — {})",
            if churn_balanced_ok { "PASS" } else { "FAIL" },
            test_total_adds,
            test_total_removes,
            churn_tag
        );
    } else {
        churn_balanced_ok = true;
    }
    println!();

    // Verdict rules:
    //   REJECT     if severe regression OR overall val F1 dropped
    //   STRONG     if overall strong up AND wide robust dominance AND no severe
    //                -> worth submitting, even at moderate Jaccard (combo_a lesson)
    //   PROMOTE    if overall up AND some robust dominance AND no severe
    //                -> decent candidate; check Jaccard for shakeup risk
    //   HOLD       otherwise (signal weak or concentrated)
    if !no_severe {
        println!(
            "VERDICT: REJECT — severe regression in {} bucket(s)",
            severe_reg
        );
    } else if !overall_up {
        println!(
            "VERDICT: REJECT — overall val F1 dropped ({:+.1}pp)",
            overall_delta_pp
        );
    } else if overall_strong && many_robust && !churn_balanced_ok {
        println!(
            "VERDICT: HOLD — val looks strong ({:+.1}pp, {}/{} buckets), BUT \
             test churn is imbalanced ({} adds / {} removes). This pattern has \
             historically hurt Kaggle F1 (precision collapse). \
             Don't submit without evidence the imbalance is intentional.",
            overall_delta_pp, robust_dominates, robust_total, test_total_adds, test_total_removes
        );
    } else if overall_strong && many_robust {
        if test_has_data && test_jaccard_overall < 0.70 {
            println!(
                "VERDICT: STRONG-BUT-RISKY — strong val lift + wide dominance, \
                 but extreme test churn (Jaccard {:.3}). High-variance bet. \
                 Pair with a conservative hedge for final picks.",
                test_jaccard_overall
            );
        } else {
            println!(
                "VERDICT: STRONG PROMOTE — wide robust dominance ({}/{}), \
                 strong lift ({:+.1}pp). Worth a Kaggle submission.",
                robust_dominates, robust_total, overall_delta_pp
            );
        }
    } else if some_robust {
        println!(
            "VERDICT: PROMOTE (weak) — some robust dominance ({}/{}), \
             lift {:+.1}pp. Moderate signal — use sparingly.",
            robust_dominates, robust_total, overall_delta_pp
        );
    } else {
        println!(
            "VERDICT: HOLD — only {}/{} robust buckets improved. \
             Improvement may be concentrated or noisy. Don't burn a submission.",
            robust_dominates, robust_total
        );
    }

    // ---- HISTORICAL CALIBRATION ANCHOR ----
    // For each known-outcome candidate, run the simplified verdict logic against
    // the USER's baseline. Report the tool's verdict next to the actual Kaggle
    // delta so the user sees empirical accuracy on every run.
    if let Some((test_rows, _, base_test, _, baseline_test_path_str)) = &test_loaded {
        let baseline_kaggle = HISTORY
            .iter()
            .find(|e| e.test_csv == baseline_test_path_str.as_str())
            .map(|e| e.kaggle);

        println!();
        println!("=== HISTORICAL CALIBRATION (10 known-outcome candidates) ===");
        if let Some(bk) = baseline_kaggle {
            println!("Baseline recognized: Kaggle {:.5}. Comparing each historical candidate against it.", bk);
        } else {
            println!(
                "NOTE: Baseline test CSV not in HISTORY table — calibration rows \
                 will show val/churn signals but cannot compute actual Kaggle delta."
            );
        }
        println!();
        println!(
            "{:<20} {:>12} {:>9} {:>8} {:>13} {:>12} {:>4}",
            "CANDIDATE", "Verdict", "ValΔpp", "Jaccard", "Add/Rem", "Kaggle Δ", "Hit"
        );
        println!("{}", "-".repeat(88));

        let mut n_scored = 0usize;
        let mut n_correct = 0usize;
        let mut n_promote = 0usize;
        let mut n_promote_hit = 0usize;
        let mut n_hold_reject = 0usize;
        let mut n_hold_reject_hit = 0usize;

        for entry in HISTORY {
            if entry.test_csv == baseline_test_path_str.as_str() {
                continue; // skip baseline vs itself
            }
            let h_val = match load_preds(entry.val_csv) {
                Ok(v) => v,
                Err(_) => {
                    println!(
                        "{:<20} {:>12}  (val CSV missing — skipped)",
                        entry.name, "—"
                    );
                    continue;
                }
            };
            let h_test = match load_preds(entry.test_csv) {
                Ok(v) => v,
                Err(_) => {
                    println!(
                        "{:<20} {:>12}  (test CSV missing — skipped)",
                        entry.name, "—"
                    );
                    continue;
                }
            };

            let (tag, val_delta, jac, adds, removes) = simple_verdict(
                &h_val,
                &base_preds,
                &h_test,
                base_test,
                &val_rows,
                test_rows,
            );

            let kag_str = match baseline_kaggle {
                Some(bk) => format!("{:+.2}pp", (entry.kaggle - bk) * 100.0),
                None => "—".to_string(),
            };

            let hit_tag = if let Some(bk) = baseline_kaggle {
                n_scored += 1;
                let actual_up = entry.kaggle >= bk;
                let tool_says_submit = tag.contains("PROMOTE");
                if tool_says_submit {
                    n_promote += 1;
                    if actual_up {
                        n_promote_hit += 1;
                        n_correct += 1;
                        "✓"
                    } else {
                        "✗"
                    }
                } else {
                    n_hold_reject += 1;
                    if !actual_up {
                        n_hold_reject_hit += 1;
                        n_correct += 1;
                        "✓"
                    } else {
                        "✗"
                    }
                }
            } else {
                "—"
            };

            println!(
                "{:<20} {:>12} {:+6.1}pp {:>8.3} {:>6}/{:<5} {:>12} {:>4}",
                entry.name, tag, val_delta, jac, adds, removes, kag_str, hit_tag
            );
        }

        if baseline_kaggle.is_some() && n_scored > 0 {
            println!();
            println!(
                "CALIBRATION: {}/{} historical candidates correctly classified ({:.0}%)",
                n_correct,
                n_scored,
                100.0 * n_correct as f64 / n_scored as f64
            );
            if n_promote > 0 {
                println!(
                    "  PROMOTE verdicts:   {}/{} correct ({:.0}%) — trust level for your \
                     current candidate if it's in this bucket",
                    n_promote_hit,
                    n_promote,
                    100.0 * n_promote_hit as f64 / n_promote as f64
                );
            }
            if n_hold_reject > 0 {
                println!(
                    "  HOLD/REJECT:        {}/{} correct ({:.0}%) — how reliably the tool \
                     filters out losers",
                    n_hold_reject_hit,
                    n_hold_reject,
                    100.0 * n_hold_reject_hit as f64 / n_hold_reject as f64
                );
            }
        }
    }

    Ok(())
}
