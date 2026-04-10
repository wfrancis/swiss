/// Cross-validation selector sweep on judged train bundles.
///
/// Loads ALL judged train chunks, splits into K folds, and for each fold:
///   - trains selector params on K-1 folds (random search)
///   - evaluates on the held-out fold
///
/// Outputs: per-fold F1, mean CV F1, best config, and optionally applies
/// the best config to val/test judged bundles for submission.
///
/// Usage:
///   cv_sweep <train_bundles_dir> <train_gold_csv> [options]
///
///   --folds 5            Number of CV folds
///   --iterations 50000   Random search iterations per fold
///   --seed 42            RNG seed
///   --val-judged PATH    Apply best config to val bundles
///   --test-judged PATH   Apply best config to test bundles
///   --val-output PATH    Write val predictions CSV
///   --test-output PATH   Write test predictions CSV
///   --config-output PATH Write best config JSON
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ConfigSnapshot {
    min_output: usize,
    max_output: usize,
    court_fraction: f64,
    min_courts_if_any: usize,
    must_keep_confidence: f64,
}

#[derive(Debug, Deserialize)]
struct Row {
    query_id: String,
    #[serde(default)]
    gold_citations: String,
}

#[derive(Debug, Deserialize, Clone)]
struct Candidate {
    citation: String,
    kind: String,
    #[allow(dead_code)]
    raw_score: f64,
    final_score: f64,
    judge_label: Option<String>,
    judge_confidence: f64,
    auto_bucket: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Bundle {
    query_id: String,
    estimated_count: usize,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    config: ConfigSnapshot,
    rows: Vec<Row>,
    bundles: Vec<Bundle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelectorConfig {
    min_output: usize,
    max_output: usize,
    court_fraction: f64,
    min_courts_if_any: usize,
    must_keep_confidence: f64,
}

fn judge_label(c: &Candidate) -> &str {
    c.judge_label.as_deref().unwrap_or("reject")
}

fn label_bonus(c: &Candidate) -> f64 {
    match judge_label(c) {
        "must_include" => 2.0,
        "plausible" => 1.0,
        _ => 0.0,
    }
}

fn cmp_desc_f64(a: f64, b: f64) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

fn python_round(value: f64) -> usize {
    let lower = value.floor();
    let fraction = value - lower;
    let epsilon = 1e-9;
    if fraction < 0.5 - epsilon {
        lower as usize
    } else if fraction > 0.5 + epsilon {
        (lower + 1.0) as usize
    } else {
        let lower_int = lower as usize;
        if lower_int % 2 == 0 { lower_int } else { lower_int + 1 }
    }
}

fn candidate_cmp(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    cmp_desc_f64(label_bonus(a), label_bonus(b))
        .then_with(|| cmp_desc_f64(a.judge_confidence, b.judge_confidence))
        .then_with(|| cmp_desc_f64(a.final_score, b.final_score))
        .then_with(|| a.citation.cmp(&b.citation))
}

fn select_bundle<'a>(bundle: &'a Bundle, cfg: &SelectorConfig) -> Vec<&'a Candidate> {
    let cfg_snap = ConfigSnapshot {
        min_output: cfg.min_output,
        max_output: cfg.max_output,
        court_fraction: cfg.court_fraction,
        min_courts_if_any: cfg.min_courts_if_any,
        must_keep_confidence: cfg.must_keep_confidence,
    };

    let mut must_cands: Vec<&Candidate> = bundle.candidates.iter()
        .filter(|c| judge_label(c) == "must_include").collect();
    let mut plausible_cands: Vec<&Candidate> = bundle.candidates.iter()
        .filter(|c| judge_label(c) == "plausible").collect();
    must_cands.sort_by(|a, b| candidate_cmp(a, b));
    plausible_cands.sort_by(|a, b| candidate_cmp(a, b));

    let mut locked_keep: Vec<&Candidate> = must_cands.iter().copied()
        .filter(|c| c.auto_bucket.as_deref() == Some("auto_keep")
            || c.judge_confidence >= cfg_snap.must_keep_confidence)
        .collect();
    locked_keep.sort_by(|a, b| candidate_cmp(a, b));

    let mut target = bundle.estimated_count.max(locked_keep.len());
    target = target.max(cfg_snap.min_output);
    target = target.min(bundle.estimated_count + 8);
    target = target.min(cfg_snap.max_output);

    let mut selected: Vec<&Candidate> = Vec::new();
    let mut ids: HashSet<&str> = HashSet::new();

    for c in &locked_keep {
        if selected.len() >= target { break; }
        if ids.insert(c.citation.as_str()) { selected.push(c); }
    }

    // Court filling
    let mut pos_courts: Vec<&Candidate> = must_cands.iter()
        .chain(plausible_cands.iter()).copied()
        .filter(|c| c.kind == "court" && !ids.contains(c.citation.as_str()))
        .collect();
    pos_courts.sort_by(|a, b| candidate_cmp(a, b));

    let mut n_courts = selected.iter().filter(|c| c.kind == "court").count();
    if !pos_courts.is_empty() {
        let court_target = cfg_snap.min_courts_if_any
            .max(python_round(target as f64 * cfg_snap.court_fraction));
        while let Some(c) = pos_courts.first().copied() {
            if n_courts >= court_target || selected.len() >= target { break; }
            pos_courts.remove(0);
            if ids.insert(c.citation.as_str()) { selected.push(c); n_courts += 1; }
        }
    }

    // Fill remaining
    let mut remaining: Vec<&Candidate> = must_cands.iter()
        .chain(plausible_cands.iter()).copied()
        .filter(|c| !ids.contains(c.citation.as_str()))
        .collect();
    remaining.sort_by(|a, b| candidate_cmp(a, b));

    for c in remaining {
        if selected.len() >= target { break; }
        if ids.insert(c.citation.as_str()) { selected.push(c); }
    }

    selected
}

fn macro_f1(predictions: &HashMap<String, BTreeSet<String>>, gold_map: &HashMap<String, HashSet<String>>) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for (qid, gold) in gold_map {
        let pred = predictions.get(qid).cloned().unwrap_or_default();
        let pred_set: HashSet<&str> = pred.iter().map(String::as_str).collect();
        let gold_set: HashSet<&str> = gold.iter().map(String::as_str).collect();
        let tp = pred_set.intersection(&gold_set).count() as f64;
        let p = if pred_set.is_empty() { 0.0 } else { tp / pred_set.len() as f64 };
        let r = if gold_set.is_empty() { 0.0 } else { tp / gold_set.len() as f64 };
        let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        total += f1;
        count += 1;
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

fn evaluate_on_bundles(
    bundles: &[&Bundle],
    gold_map: &HashMap<String, HashSet<String>>,
    cfg: &SelectorConfig,
) -> f64 {
    let mut predictions: HashMap<String, BTreeSet<String>> = HashMap::new();
    for bundle in bundles {
        let selected = select_bundle(bundle, cfg);
        let cites: BTreeSet<String> = selected.iter().map(|c| c.citation.clone()).collect();
        predictions.insert(bundle.query_id.clone(), cites);
    }
    macro_f1(&predictions, gold_map)
}

fn random_config(rng: &mut StdRng) -> SelectorConfig {
    let court_fracs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];
    let confs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.86, 0.90, 0.95];
    let max_outs = [15, 20, 25, 30, 35, 40, 45, 50, 60];
    let min_outs = [4, 6, 8, 10, 12, 15, 18, 20];
    let min_courts_vals = [1, 2, 3, 4, 5, 6, 8];

    let max_output = *max_outs.choose(rng).unwrap();
    let min_output = *min_outs.choose(rng).unwrap();
    let min_output = min_output.min(max_output - 1);

    SelectorConfig {
        court_fraction: *court_fracs.choose(rng).unwrap(),
        must_keep_confidence: *confs.choose(rng).unwrap(),
        max_output,
        min_output,
        min_courts_if_any: *min_courts_vals.choose(rng).unwrap(),
    }
}

fn write_predictions_csv(
    predictions: &HashMap<String, BTreeSet<String>>,
    path: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_path(path)?;
    wtr.write_record(["query_id", "predicted_citations"])?;
    let mut qids: Vec<&String> = predictions.keys().collect();
    qids.sort();
    for qid in qids {
        let cites = predictions.get(qid)
            .map(|s| s.iter().cloned().collect::<Vec<_>>().join(";"))
            .unwrap_or_default();
        wtr.write_record([qid.as_str(), cites.as_str()])?;
    }
    wtr.flush()?;
    Ok(())
}

fn load_gold(path: &PathBuf) -> Result<HashMap<String, HashSet<String>>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut gold_map: HashMap<String, HashSet<String>> = HashMap::new();
    for result in rdr.deserialize::<HashMap<String, String>>() {
        let row = result?;
        let qid = row.get("query_id").cloned().unwrap_or_default();
        let cites: HashSet<String> = row.get("gold_citations")
            .map(|v| v.split(';').filter(|s| !s.is_empty()).map(String::from).collect())
            .unwrap_or_default();
        gold_map.insert(qid, cites);
    }
    Ok(gold_map)
}

fn load_all_train_bundles(dir: &PathBuf) -> Result<(Vec<Bundle>, ConfigSnapshot), Box<dyn Error>> {
    let mut all_bundles: Vec<Bundle> = Vec::new();
    let mut config: Option<ConfigSnapshot> = None;

    // Load individual chunk JSONs
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.file_name()
            .map(|n| n.to_string_lossy().starts_with("train_"))
            .unwrap_or(false)
        {
            let json_path = path.join("judged_bundles.json");
            if json_path.exists() {
                paths.push(json_path);
            }
        }
    }
    paths.sort();

    eprintln!("Loading {} chunk artifacts...", paths.len());
    for path in &paths {
        let text = fs::read_to_string(path)?;
        let artifact: Artifact = serde_json::from_str(&text)?;
        if config.is_none() {
            config = Some(artifact.config);
        }
        all_bundles.extend(artifact.bundles);
    }

    let cfg = config.unwrap_or(ConfigSnapshot {
        min_output: 10,
        max_output: 40,
        court_fraction: 0.25,
        min_courts_if_any: 4,
        must_keep_confidence: 0.86,
    });

    eprintln!("Loaded {} total train bundles", all_bundles.len());
    Ok((all_bundles, cfg))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cv_sweep <train_bundles_dir> <train_gold_csv> [options]");
        eprintln!("Options:");
        eprintln!("  --folds N          (default: 5)");
        eprintln!("  --iterations N     (default: 50000)");
        eprintln!("  --seed N           (default: 42)");
        eprintln!("  --val-judged PATH");
        eprintln!("  --test-judged PATH");
        eprintln!("  --val-output PATH");
        eprintln!("  --test-output PATH");
        eprintln!("  --config-output PATH");
        std::process::exit(1);
    }

    let bundles_dir = PathBuf::from(&args[1]);
    let gold_csv = PathBuf::from(&args[2]);

    let mut n_folds: usize = 5;
    let mut iterations: usize = 50000;
    let mut seed: u64 = 42;
    let mut val_judged: Option<PathBuf> = None;
    let mut test_judged: Option<PathBuf> = None;
    let mut val_output: Option<PathBuf> = None;
    let mut test_output: Option<PathBuf> = None;
    let mut config_output: Option<PathBuf> = None;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--folds" => { i += 1; n_folds = args[i].parse()?; }
            "--iterations" => { i += 1; iterations = args[i].parse()?; }
            "--seed" => { i += 1; seed = args[i].parse()?; }
            "--val-judged" => { i += 1; val_judged = Some(PathBuf::from(&args[i])); }
            "--test-judged" => { i += 1; test_judged = Some(PathBuf::from(&args[i])); }
            "--val-output" => { i += 1; val_output = Some(PathBuf::from(&args[i])); }
            "--test-output" => { i += 1; test_output = Some(PathBuf::from(&args[i])); }
            "--config-output" => { i += 1; config_output = Some(PathBuf::from(&args[i])); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    // Load data
    let gold_map = load_gold(&gold_csv)?;
    let (all_bundles, _default_config) = load_all_train_bundles(&bundles_dir)?;

    // Filter to bundles that have gold labels
    let train_bundles: Vec<&Bundle> = all_bundles.iter()
        .filter(|b| gold_map.contains_key(&b.query_id))
        .collect();
    let n = train_bundles.len();
    eprintln!("{} train bundles with gold labels", n);

    if n < n_folds {
        eprintln!("Not enough bundles ({}) for {} folds", n, n_folds);
        std::process::exit(1);
    }

    // Shuffle and split into folds
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let fold_size = n / n_folds;
    let folds: Vec<Vec<usize>> = (0..n_folds).map(|f| {
        let start = f * fold_size;
        let end = if f == n_folds - 1 { n } else { start + fold_size };
        indices[start..end].to_vec()
    }).collect();

    eprintln!("\n=== {}-FOLD CROSS-VALIDATION ({} iterations/fold) ===\n", n_folds, iterations);

    let mut fold_best_configs: Vec<SelectorConfig> = Vec::new();
    let mut fold_test_f1s: Vec<f64> = Vec::new();
    let mut global_best_f1 = 0.0f64;
    let mut global_best_config: Option<SelectorConfig> = None;

    for fold_idx in 0..n_folds {
        let test_indices: HashSet<usize> = folds[fold_idx].iter().copied().collect();
        let fold_train: Vec<&Bundle> = (0..n).filter(|i| !test_indices.contains(i))
            .map(|i| train_bundles[i]).collect();
        let fold_test: Vec<&Bundle> = folds[fold_idx].iter()
            .map(|&i| train_bundles[i]).collect();

        let fold_train_gold: HashMap<String, HashSet<String>> = fold_train.iter()
            .filter_map(|b| gold_map.get(&b.query_id).map(|g| (b.query_id.clone(), g.clone())))
            .collect();

        let fold_test_gold: HashMap<String, HashSet<String>> = fold_test.iter()
            .filter_map(|b| gold_map.get(&b.query_id).map(|g| (b.query_id.clone(), g.clone())))
            .collect();

        let mut best_train_f1 = 0.0f64;
        let mut best_config = random_config(&mut rng);

        for iter in 0..iterations {
            let cfg = random_config(&mut rng);
            let f1 = evaluate_on_bundles(&fold_train, &fold_train_gold, &cfg);
            if f1 > best_train_f1 {
                best_train_f1 = f1;
                best_config = cfg;
            }
            if (iter + 1) % 10000 == 0 {
                eprint!("  fold {}: {}/{} iters, best train F1={:.6}\r",
                    fold_idx + 1, iter + 1, iterations, best_train_f1);
            }
        }

        let fold_test_f1 = evaluate_on_bundles(&fold_test, &fold_test_gold, &best_config);

        eprintln!("  Fold {}/{}: train F1={:.6}, TEST F1={:.6}  cfg={:?}",
            fold_idx + 1, n_folds, best_train_f1, fold_test_f1, best_config);

        fold_best_configs.push(best_config.clone());
        fold_test_f1s.push(fold_test_f1);

        // Track global best by test F1
        if fold_test_f1 > global_best_f1 {
            global_best_f1 = fold_test_f1;
            global_best_config = Some(best_config);
        }
    }

    let mean_cv_f1: f64 = fold_test_f1s.iter().sum::<f64>() / fold_test_f1s.len() as f64;
    let std_cv_f1 = {
        let var = fold_test_f1s.iter().map(|f| (f - mean_cv_f1).powi(2)).sum::<f64>() / fold_test_f1s.len() as f64;
        var.sqrt()
    };

    eprintln!("\n=== RESULTS ===");
    eprintln!("Mean CV F1: {:.6} (+/- {:.6})", mean_cv_f1, std_cv_f1);
    for (i, f1) in fold_test_f1s.iter().enumerate() {
        eprintln!("  Fold {}: {:.6}", i + 1, f1);
    }

    // Also evaluate default config for comparison
    let default_cfg = SelectorConfig {
        min_output: 10, max_output: 40, court_fraction: 0.25,
        min_courts_if_any: 4, must_keep_confidence: 0.86,
    };
    let default_f1 = evaluate_on_bundles(&train_bundles, &gold_map, &default_cfg);
    eprintln!("Default config full-train F1: {:.6}", default_f1);

    let best_cfg = global_best_config.unwrap_or(default_cfg);
    eprintln!("\nBest config: {:?}", best_cfg);

    // Print as JSON to stdout
    println!("{}", serde_json::to_string_pretty(&best_cfg)?);

    // Save config
    if let Some(path) = &config_output {
        fs::write(path, serde_json::to_string_pretty(&best_cfg)?)?;
        eprintln!("Saved config to {}", path.display());
    }

    // Apply best config to val/test if provided
    if let Some(val_path) = &val_judged {
        let text = fs::read_to_string(val_path)?;
        let artifact: Artifact = serde_json::from_str(&text)?;
        let mut preds: HashMap<String, BTreeSet<String>> = HashMap::new();
        for bundle in &artifact.bundles {
            let selected = select_bundle(bundle, &best_cfg);
            preds.insert(bundle.query_id.clone(), selected.iter().map(|c| c.citation.clone()).collect());
        }

        // Evaluate on val gold if available
        let val_gold: HashMap<String, HashSet<String>> = artifact.rows.iter()
            .filter(|r| !r.gold_citations.is_empty())
            .map(|r| {
                let cites: HashSet<String> = r.gold_citations.split(';')
                    .filter(|s| !s.is_empty()).map(String::from).collect();
                (r.query_id.clone(), cites)
            }).collect();
        if !val_gold.is_empty() {
            let val_f1 = macro_f1(&preds, &val_gold);
            eprintln!("Val F1 with best config: {:.6}", val_f1);
        }

        if let Some(out) = &val_output {
            write_predictions_csv(&preds, out)?;
            eprintln!("Wrote val predictions to {}", out.display());
        }
    }

    if let Some(test_path) = &test_judged {
        let text = fs::read_to_string(test_path)?;
        let artifact: Artifact = serde_json::from_str(&text)?;
        let mut preds: HashMap<String, BTreeSet<String>> = HashMap::new();
        for bundle in &artifact.bundles {
            let selected = select_bundle(bundle, &best_cfg);
            preds.insert(bundle.query_id.clone(), selected.iter().map(|c| c.citation.clone()).collect());
        }
        if let Some(out) = &test_output {
            write_predictions_csv(&preds, out)?;
            eprintln!("Wrote test predictions to {}", out.display());
        }
    }

    Ok(())
}
