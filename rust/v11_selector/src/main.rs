use serde::Deserialize;
use std::cmp::Ordering;
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

fn judge_label(candidate: &Candidate) -> &str {
    candidate.judge_label.as_deref().unwrap_or("reject")
}

fn label_bonus(candidate: &Candidate) -> f64 {
    match judge_label(candidate) {
        "must_include" => 2.0,
        "plausible" => 1.0,
        _ => 0.0,
    }
}

fn cmp_desc_f64(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
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
        if lower_int % 2 == 0 {
            lower_int
        } else {
            lower_int + 1
        }
    }
}

fn candidate_cmp(left: &Candidate, right: &Candidate) -> Ordering {
    cmp_desc_f64(label_bonus(left), label_bonus(right))
        .then_with(|| cmp_desc_f64(left.judge_confidence, right.judge_confidence))
        .then_with(|| cmp_desc_f64(left.final_score, right.final_score))
        .then_with(|| left.citation.cmp(&right.citation))
}

fn select_bundle<'a>(bundle: &'a Bundle, config: &ConfigSnapshot) -> Vec<&'a Candidate> {
    let mut must_candidates: Vec<&Candidate> = bundle
        .candidates
        .iter()
        .filter(|candidate| judge_label(candidate) == "must_include")
        .collect();
    let mut plausible_candidates: Vec<&Candidate> = bundle
        .candidates
        .iter()
        .filter(|candidate| judge_label(candidate) == "plausible")
        .collect();

    must_candidates.sort_by(|left, right| candidate_cmp(left, right));
    plausible_candidates.sort_by(|left, right| candidate_cmp(left, right));

    let mut locked_keep: Vec<&Candidate> = must_candidates
        .iter()
        .copied()
        .filter(|candidate| {
            candidate.auto_bucket.as_deref() == Some("auto_keep")
                || candidate.judge_confidence >= config.must_keep_confidence
        })
        .collect();
    locked_keep.sort_by(|left, right| candidate_cmp(left, right));

    let mut target = bundle.estimated_count.max(locked_keep.len());
    target = target.max(config.min_output);
    target = target.min(bundle.estimated_count + 8);
    target = target.min(config.max_output);

    let mut selected: Vec<&Candidate> = Vec::new();
    let mut selected_ids: HashSet<&str> = HashSet::new();

    for candidate in locked_keep {
        if selected.len() >= target {
            break;
        }
        if selected_ids.insert(candidate.citation.as_str()) {
            selected.push(candidate);
        }
    }

    let mut positive_courts: Vec<&Candidate> = must_candidates
        .iter()
        .chain(plausible_candidates.iter())
        .copied()
        .filter(|candidate| candidate.kind == "court" && !selected_ids.contains(candidate.citation.as_str()))
        .collect();
    positive_courts.sort_by(|left, right| candidate_cmp(left, right));

    let mut selected_courts = selected.iter().filter(|candidate| candidate.kind == "court").count();
    if !positive_courts.is_empty() {
        let soft_court_target = config
            .min_courts_if_any
            .max(python_round(target as f64 * config.court_fraction));
        while let Some(candidate) = positive_courts.first().copied() {
            if selected_courts >= soft_court_target || selected.len() >= target {
                break;
            }
            positive_courts.remove(0);
            if selected_ids.insert(candidate.citation.as_str()) {
                selected.push(candidate);
                selected_courts += 1;
            }
        }
    }

    let mut remaining_positive: Vec<&Candidate> = must_candidates
        .iter()
        .chain(plausible_candidates.iter())
        .copied()
        .filter(|candidate| !selected_ids.contains(candidate.citation.as_str()))
        .collect();
    remaining_positive.sort_by(|left, right| candidate_cmp(left, right));

    for candidate in remaining_positive {
        if selected.len() >= target {
            break;
        }
        if selected_ids.insert(candidate.citation.as_str()) {
            selected.push(candidate);
        }
    }

    selected
}

fn evaluate_predictions(
    predictions: &HashMap<String, BTreeSet<String>>,
    rows: &[Row],
) -> Option<f64> {
    if !rows.iter().any(|row| !row.gold_citations.is_empty()) {
        return None;
    }

    let mut total_f1 = 0.0;
    let mut count = 0usize;
    for row in rows {
        let gold: HashSet<&str> = row
            .gold_citations
            .split(';')
            .filter(|value| !value.is_empty())
            .collect();
        let pred = predictions.get(&row.query_id).cloned().unwrap_or_default();
        let pred_set: HashSet<&str> = pred.iter().map(String::as_str).collect();
        let tp = pred_set.intersection(&gold).count() as f64;
        let precision = if pred_set.is_empty() { 0.0 } else { tp / pred_set.len() as f64 };
        let recall = if gold.is_empty() { 0.0 } else { tp / gold.len() as f64 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        total_f1 += f1;
        count += 1;
    }

    if count == 0 {
        None
    } else {
        Some(total_f1 / count as f64)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or("usage: v11_selector <judged_json> <output_csv>")?);
    let output = PathBuf::from(args.next().ok_or("usage: v11_selector <judged_json> <output_csv>")?);

    let artifact: Artifact = serde_json::from_str(&fs::read_to_string(&input)?)?;
    let mut predictions: HashMap<String, BTreeSet<String>> = HashMap::new();

    for bundle in &artifact.bundles {
        let selected = select_bundle(bundle, &artifact.config);
        let auto_keep = bundle
            .candidates
            .iter()
            .filter(|candidate| candidate.auto_bucket.as_deref() == Some("auto_keep"))
            .count();
        let judge_count = bundle
            .candidates
            .iter()
            .filter(|candidate| candidate.auto_bucket.is_none())
            .count();
        let must_count = bundle
            .candidates
            .iter()
            .filter(|candidate| judge_label(candidate) == "must_include")
            .count();
        let plausible_count = bundle
            .candidates
            .iter()
            .filter(|candidate| judge_label(candidate) == "plausible")
            .count();
        let reject_count = bundle
            .candidates
            .iter()
            .filter(|candidate| judge_label(candidate) == "reject")
            .count();

        let citations: BTreeSet<String> = selected.iter().map(|candidate| candidate.citation.clone()).collect();
        println!(
            "  {}: auto_keep={}, judge={}, selected={} (must={}, plausible={}, reject={})",
            bundle.query_id,
            auto_keep,
            judge_count,
            citations.len(),
            must_count,
            plausible_count,
            reject_count
        );
        predictions.insert(bundle.query_id.clone(), citations);
    }

    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_path(&output)?;
    writer.write_record(["query_id", "predicted_citations"])?;
    let mut query_ids: Vec<&String> = predictions.keys().collect();
    query_ids.sort();
    for query_id in query_ids {
        let citations = predictions
            .get(query_id)
            .map(|items| items.iter().cloned().collect::<Vec<_>>().join(";"))
            .unwrap_or_default();
        writer.write_record([query_id.as_str(), citations.as_str()])?;
    }
    writer.flush()?;

    if let Some(macro_f1) = evaluate_predictions(&predictions, &artifact.rows) {
        println!("\n=== V11 RUST MACRO F1: {:.4} ({:.2}%) ===", macro_f1, macro_f1 * 100.0);
    }

    let avg_predictions = if predictions.is_empty() {
        0.0
    } else {
        let total: usize = predictions.values().map(BTreeSet::len).sum();
        total as f64 / predictions.len() as f64
    };
    println!("Saved to {}", output.display());
    println!("Average predictions: {:.1}", avg_predictions);
    Ok(())
}
