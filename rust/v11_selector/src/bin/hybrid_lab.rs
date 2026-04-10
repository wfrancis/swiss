use csv::{ReaderBuilder, WriterBuilder};
use rand::prelude::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
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
    raw_score: f64,
    final_score: f64,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    gpt_full_freq: usize,
    #[serde(default)]
    is_explicit: bool,
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
struct HybridConfig {
    v7_bonus: f64,
    v11_bonus: f64,
    both_bonus: f64,
    auto_keep_bonus: f64,
    explicit_bonus: f64,
    must_bonus: f64,
    plausible_bonus: f64,
    reject_penalty: f64,
    conf_weight: f64,
    final_weight: f64,
    raw_weight: f64,
    gpt_freq_weight: f64,
    source_count_weight: f64,
    dense_bonus: f64,
    bm25_bonus: f64,
    gpt_case_bonus: f64,
    cocitation_penalty: f64,
    court_dense_bonus: f64,
    law_bonus: f64,
    court_bonus: f64,
    exact_train_weight: f64,
    dense_train_weight: f64,
    law_base_train_weight: f64,
    dense_law_base_train_weight: f64,
    #[serde(default)]
    dense100_exact_weight: f64,
    #[serde(default)]
    dense12_exact_weight: f64,
    #[serde(default)]
    sparse79_exact_weight: f64,
    #[serde(default)]
    dense100_law_base_weight: f64,
    #[serde(default)]
    dense12_law_base_weight: f64,
    #[serde(default)]
    sparse79_law_base_weight: f64,
    court_dense_only_penalty: f64,
    single_source_penalty: f64,
    target_mult: f64,
    target_bias: i32,
    min_output: usize,
    max_output: usize,
    court_cap_frac: f64,
    max_law_per_base: usize,
    max_court_per_base: usize,
}

#[derive(Debug, Default)]
struct PriorChannel {
    exact: HashMap<String, usize>,
    law_base: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct HybridCandidate {
    citation: String,
    kind: String,
    in_v7: bool,
    in_v11: bool,
    score: f64,
}

#[derive(Debug, Clone)]
struct EvalResult {
    macro_f1: Option<f64>,
    per_query_f1: Vec<f64>,
    predictions: HashMap<String, BTreeSet<String>>,
    avg_prediction_count: f64,
    avg_court_fraction: f64,
}

#[derive(Debug, Clone)]
struct CandidateResult {
    objective: f64,
    val_result: EvalResult,
    test_result: Option<EvalResult>,
}

#[derive(Debug, Default)]
struct TrainPriors {
    exact_all: HashMap<String, usize>,
    exact_dense: HashMap<String, usize>,
    law_base_all: HashMap<String, usize>,
    law_base_dense: HashMap<String, usize>,
    dense100: PriorChannel,
    dense12: PriorChannel,
    sparse79: PriorChannel,
}

#[derive(Debug, Deserialize)]
struct ExpansionRow {
    #[serde(default)]
    specific_articles: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CaseCitationRow {
    #[serde(default)]
    expanded: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FullCitationRow {
    #[serde(default)]
    law_citations: Vec<String>,
    #[serde(default)]
    court_citations: Vec<String>,
}

fn judge_label(candidate: &Candidate) -> &str {
    candidate.judge_label.as_deref().unwrap_or("reject")
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

fn cmp_desc_f64(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn label_bonus(candidate: &Candidate) -> f64 {
    match judge_label(candidate) {
        "must_include" => 2.0,
        "plausible" => 1.0,
        _ => 0.0,
    }
}

fn candidate_cmp(left: &Candidate, right: &Candidate) -> Ordering {
    cmp_desc_f64(label_bonus(left), label_bonus(right))
        .then_with(|| cmp_desc_f64(left.judge_confidence, right.judge_confidence))
        .then_with(|| cmp_desc_f64(left.final_score, right.final_score))
        .then_with(|| left.citation.cmp(&right.citation))
}

fn select_v11_bundle<'a>(bundle: &'a Bundle, config: &ConfigSnapshot) -> Vec<&'a Candidate> {
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

fn parse_kind(citation: &str) -> String {
    if citation.starts_with("Art.") {
        "law".to_string()
    } else {
        "court".to_string()
    }
}

fn law_base(citation: &str) -> Option<String> {
    if !citation.starts_with("Art.") {
        return None;
    }
    let statute = citation.split_whitespace().last()?;
    let article = citation.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
    Some(format!("{article} {statute}"))
}

fn court_base(citation: &str) -> Option<String> {
    if citation.starts_with("Art.") {
        return None;
    }
    Some(
        citation
            .split(" E. ")
            .next()
            .unwrap_or(citation)
            .trim()
            .to_string(),
    )
}

fn log_count(map: &HashMap<String, usize>, key: Option<String>) -> f64 {
    key.and_then(|key| map.get(&key).copied())
        .map(|count| (count as f64 + 1.0).ln())
        .unwrap_or(0.0)
}

fn add_prior_citation(priors: &mut TrainPriors, citation: &str, dense: bool) {
    if citation.is_empty() {
        return;
    }
    *priors.exact_all.entry(citation.to_string()).or_insert(0) += 1;
    if dense {
        *priors.exact_dense.entry(citation.to_string()).or_insert(0) += 1;
    }
    if let Some(base) = law_base(citation) {
        *priors.law_base_all.entry(base.clone()).or_insert(0) += 1;
        if dense {
            *priors.law_base_dense.entry(base).or_insert(0) += 1;
        }
    }
}

fn add_channel_citation(channel: &mut PriorChannel, citation: &str) {
    if citation.is_empty() {
        return;
    }
    *channel.exact.entry(citation.to_string()).or_insert(0) += 1;
    if let Some(base) = law_base(citation) {
        *channel.law_base.entry(base).or_insert(0) += 1;
    }
}

fn load_query_ids(path: &PathBuf) -> HashSet<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| line.to_string())
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default()
}

fn load_train_priors(train_csv: &PathBuf) -> Result<TrainPriors, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().from_path(train_csv)?;
    let mut priors = TrainPriors::default();
    let mut dense_query_ids: HashSet<String> = HashSet::new();
    let mut gold_counts: HashMap<String, usize> = HashMap::new();
    for row in reader.deserialize::<HashMap<String, String>>() {
        let row = row?;
        let query_id = row.get("query_id").cloned().unwrap_or_default();
        let citations = row
            .get("gold_citations")
            .map(|value| value.split(';').filter(|value| !value.is_empty()).collect::<Vec<_>>())
            .unwrap_or_default();
        let dense = citations.len() >= 10;
        if !query_id.is_empty() {
            gold_counts.insert(query_id.clone(), citations.len());
        }
        if dense && !query_id.is_empty() {
            dense_query_ids.insert(query_id);
        }
        for citation in citations {
            add_prior_citation(&mut priors, citation, dense);
        }
    }

    let base_dir = train_csv
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let repo_root = base_dir.parent().map(PathBuf::from).unwrap_or(base_dir);
    let precompute_dir = repo_root.clone();
    let artifacts_dir = repo_root.join("artifacts");

    let dense100_ids = load_query_ids(&artifacts_dir.join("dense_train_qids_100.txt"));
    let stage2_ids = load_query_ids(&artifacts_dir.join("dense_train_qids_200_stage2.txt"));
    let dense12_ids: HashSet<String> = stage2_ids
        .iter()
        .filter(|query_id| gold_counts.get(*query_id).copied().unwrap_or(0) >= 10)
        .cloned()
        .collect();
    let sparse79_ids: HashSet<String> = stage2_ids
        .iter()
        .filter(|query_id| {
            let count = gold_counts.get(*query_id).copied().unwrap_or(0);
            (7..=9).contains(&count)
        })
        .cloned()
        .collect();

    let full_path = precompute_dir.join("precompute").join("train_full_citations_v2.json");
    if let Ok(text) = fs::read_to_string(&full_path) {
        if let Ok(rows) = serde_json::from_str::<HashMap<String, FullCitationRow>>(&text) {
            for (query_id, row) in rows {
                for citation in row
                    .law_citations
                    .iter()
                    .chain(row.court_citations.iter())
                {
                    if dense100_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense100, citation);
                    } else if dense12_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense12, citation);
                    } else if sparse79_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.sparse79, citation);
                    }
                }
            }
        }
    }

    let case_path = precompute_dir.join("precompute").join("train_case_citations.json");
    if let Ok(text) = fs::read_to_string(&case_path) {
        if let Ok(rows) = serde_json::from_str::<HashMap<String, CaseCitationRow>>(&text) {
            for (query_id, row) in rows {
                for citation in row.expanded {
                    if dense100_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense100, &citation);
                    } else if dense12_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense12, &citation);
                    } else if sparse79_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.sparse79, &citation);
                    }
                }
            }
        }
    }

    let expansions_path = precompute_dir.join("precompute").join("train_query_expansions.json");
    if let Ok(text) = fs::read_to_string(&expansions_path) {
        if let Ok(rows) = serde_json::from_str::<HashMap<String, ExpansionRow>>(&text) {
            for (query_id, row) in rows {
                for citation in row.specific_articles {
                    if dense100_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense100, &citation);
                    } else if dense12_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.dense12, &citation);
                    } else if sparse79_ids.contains(&query_id) {
                        add_channel_citation(&mut priors.sparse79, &citation);
                    }
                }
            }
        }
    }

    Ok(priors)
}

fn parse_prediction_csv(path: &PathBuf) -> Result<HashMap<String, BTreeSet<String>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut predictions: HashMap<String, BTreeSet<String>> = HashMap::new();
    for row in reader.deserialize::<HashMap<String, String>>() {
        let row = row?;
        let query_id = row.get("query_id").ok_or("missing query_id")?.to_string();
        let citations_key = if row.contains_key("predicted_citations") {
            "predicted_citations"
        } else {
            "citations"
        };
        let citations = row
            .get(citations_key)
            .map(|value| {
                value
                    .split(';')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.to_string())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        predictions.insert(query_id, citations);
    }
    Ok(predictions)
}

fn f1(pred: &BTreeSet<String>, gold_citations: &str) -> f64 {
    let gold: HashSet<&str> = gold_citations
        .split(';')
        .filter(|value| !value.is_empty())
        .collect();
    let pred_set: HashSet<&str> = pred.iter().map(String::as_str).collect();
    let tp = pred_set.intersection(&gold).count() as f64;
    let precision = if pred_set.is_empty() { 0.0 } else { tp / pred_set.len() as f64 };
    let recall = if gold.is_empty() { 0.0 } else { tp / gold.len() as f64 };
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn prediction_stats(predictions: &HashMap<String, BTreeSet<String>>) -> (f64, f64) {
    if predictions.is_empty() {
        return (0.0, 0.0);
    }
    let mut total_predictions = 0usize;
    let mut court_fraction_sum = 0.0;
    let mut keys: Vec<&String> = predictions.keys().collect();
    keys.sort();
    for key in keys {
        let citations = predictions.get(key).cloned().unwrap_or_default();
        total_predictions += citations.len();
        if citations.is_empty() {
            continue;
        }
        let courts = citations
            .iter()
            .filter(|citation| !citation.starts_with("Art."))
            .count();
        court_fraction_sum += courts as f64 / citations.len() as f64;
    }
    (
        total_predictions as f64 / predictions.len() as f64,
        court_fraction_sum / predictions.len() as f64,
    )
}

fn jaccard_average(
    left: &HashMap<String, BTreeSet<String>>,
    right: &HashMap<String, BTreeSet<String>>,
) -> f64 {
    if left.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    let mut keys: Vec<&String> = left.keys().collect();
    keys.sort();
    for key in keys {
        let left_set = left.get(key).cloned().unwrap_or_default();
        let right_set = right.get(key).cloned().unwrap_or_default();
        let union = left_set.union(&right_set).count();
        if union == 0 {
            score += 1.0;
        } else {
            score += left_set.intersection(&right_set).count() as f64 / union as f64;
        }
    }
    score / left.len() as f64
}

fn stddev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn low_penalty(value: f64, floor: f64, weight: f64) -> f64 {
    if value >= floor {
        0.0
    } else {
        (floor - value) * weight
    }
}

fn hybrid_score(
    cfg: &HybridConfig,
    citation: &str,
    in_v7: bool,
    in_v11: bool,
    candidate: Option<&Candidate>,
    priors: &TrainPriors,
) -> f64 {
    let law_key = law_base(citation);
    let mut score = 0.0;
    if in_v7 {
        score += cfg.v7_bonus;
    }
    if in_v11 {
        score += cfg.v11_bonus;
    }
    if in_v7 && in_v11 {
        score += cfg.both_bonus;
    }
    if let Some(candidate) = candidate {
        if candidate.auto_bucket.as_deref() == Some("auto_keep") {
            score += cfg.auto_keep_bonus;
        }
        if candidate.is_explicit {
            score += cfg.explicit_bonus;
        }
        match judge_label(candidate) {
            "must_include" => score += cfg.must_bonus,
            "plausible" => score += cfg.plausible_bonus,
            "reject" => score += cfg.reject_penalty,
            _ => {}
        }
        score += candidate.judge_confidence * cfg.conf_weight;
        score += candidate.final_score * cfg.final_weight;
        score += candidate.raw_score * cfg.raw_weight;
        score += candidate.gpt_full_freq as f64 * cfg.gpt_freq_weight;
        score += candidate.sources.len() as f64 * cfg.source_count_weight;
        if candidate.kind == "law" {
            score += cfg.law_bonus;
        } else {
            score += cfg.court_bonus;
        }
        if candidate.sources.iter().any(|source| source == "dense") {
            score += cfg.dense_bonus;
        }
        if candidate.sources.iter().any(|source| source == "bm25") {
            score += cfg.bm25_bonus;
        }
        if candidate.sources.iter().any(|source| source == "gpt_case") {
            score += cfg.gpt_case_bonus;
        }
        if candidate.sources.iter().any(|source| source == "cocitation") {
            score += cfg.cocitation_penalty;
        }
        if candidate.sources.iter().any(|source| source == "court_dense") {
            score += cfg.court_dense_bonus;
        }
        if candidate.sources.len() == 1 && candidate.sources[0] == "court_dense" {
            score += cfg.court_dense_only_penalty;
        }
        if candidate.sources.len() <= 1 {
            score += cfg.single_source_penalty;
        }
        score += cfg.exact_train_weight * log_count(&priors.exact_all, Some(citation.to_string()));
        score += cfg.dense_train_weight * log_count(&priors.exact_dense, Some(citation.to_string()));
        score += cfg.law_base_train_weight * log_count(&priors.law_base_all, law_key.clone());
        score += cfg.dense_law_base_train_weight * log_count(&priors.law_base_dense, law_key.clone());
        score += cfg.dense100_exact_weight * log_count(&priors.dense100.exact, Some(citation.to_string()));
        score += cfg.dense12_exact_weight * log_count(&priors.dense12.exact, Some(citation.to_string()));
        score += cfg.sparse79_exact_weight * log_count(&priors.sparse79.exact, Some(citation.to_string()));
        score += cfg.dense100_law_base_weight * log_count(&priors.dense100.law_base, law_key.clone());
        score += cfg.dense12_law_base_weight * log_count(&priors.dense12.law_base, law_key.clone());
        score += cfg.sparse79_law_base_weight * log_count(&priors.sparse79.law_base, law_key.clone());
    } else if parse_kind(citation) == "law" {
        score += cfg.law_bonus;
        score += cfg.exact_train_weight * log_count(&priors.exact_all, Some(citation.to_string()));
        score += cfg.dense_train_weight * log_count(&priors.exact_dense, Some(citation.to_string()));
        score += cfg.law_base_train_weight * log_count(&priors.law_base_all, law_key.clone());
        score += cfg.dense_law_base_train_weight * log_count(&priors.law_base_dense, law_key.clone());
        score += cfg.dense100_exact_weight * log_count(&priors.dense100.exact, Some(citation.to_string()));
        score += cfg.dense12_exact_weight * log_count(&priors.dense12.exact, Some(citation.to_string()));
        score += cfg.sparse79_exact_weight * log_count(&priors.sparse79.exact, Some(citation.to_string()));
        score += cfg.dense100_law_base_weight * log_count(&priors.dense100.law_base, law_key.clone());
        score += cfg.dense12_law_base_weight * log_count(&priors.dense12.law_base, law_key.clone());
        score += cfg.sparse79_law_base_weight * log_count(&priors.sparse79.law_base, law_key.clone());
    } else {
        score += cfg.court_bonus;
        score += cfg.dense100_exact_weight * log_count(&priors.dense100.exact, Some(citation.to_string()));
        score += cfg.dense12_exact_weight * log_count(&priors.dense12.exact, Some(citation.to_string()));
        score += cfg.sparse79_exact_weight * log_count(&priors.sparse79.exact, Some(citation.to_string()));
    }
    score
}

fn evaluate_config(
    artifact: &Artifact,
    v7_predictions: &HashMap<String, BTreeSet<String>>,
    v11_predictions: &HashMap<String, BTreeSet<String>>,
    cfg: &HybridConfig,
    priors: &TrainPriors,
) -> EvalResult {
    let mut macro_f1 = 0.0;
    let mut macro_count = 0usize;
    let has_gold = artifact.rows.iter().any(|row| !row.gold_citations.is_empty());
    let gold_map: HashMap<&str, &str> = artifact
        .rows
        .iter()
        .map(|row| (row.query_id.as_str(), row.gold_citations.as_str()))
        .collect();
    let mut predictions: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut per_query_f1 = Vec::new();

    for bundle in &artifact.bundles {
        let v7 = v7_predictions.get(&bundle.query_id).cloned().unwrap_or_default();
        let v11 = v11_predictions.get(&bundle.query_id).cloned().unwrap_or_default();
        let mut candidate_map: HashMap<String, HybridCandidate> = HashMap::new();

        for candidate in &bundle.candidates {
            let in_v7 = v7.contains(&candidate.citation);
            let in_v11 = v11.contains(&candidate.citation);
            let score = hybrid_score(cfg, &candidate.citation, in_v7, in_v11, Some(candidate), priors);
            candidate_map.insert(
                candidate.citation.clone(),
                HybridCandidate {
                    citation: candidate.citation.clone(),
                    kind: candidate.kind.clone(),
                    in_v7,
                    in_v11,
                    score,
                },
            );
        }

        for citation in &v7 {
            candidate_map.entry(citation.clone()).or_insert_with(|| HybridCandidate {
                citation: citation.clone(),
                kind: parse_kind(citation),
                in_v7: true,
                in_v11: v11.contains(citation),
                score: hybrid_score(cfg, citation, true, v11.contains(citation), None, priors),
            });
        }

        for citation in &v11 {
            candidate_map.entry(citation.clone()).or_insert_with(|| HybridCandidate {
                citation: citation.clone(),
                kind: parse_kind(citation),
                in_v7: v7.contains(citation),
                in_v11: true,
                score: hybrid_score(cfg, citation, v7.contains(citation), true, None, priors),
            });
        }

        let mut ranked: Vec<HybridCandidate> = candidate_map.into_values().collect();
        ranked.sort_by(|left, right| {
            cmp_desc_f64(left.score, right.score)
                .then_with(|| right.in_v7.cmp(&left.in_v7))
                .then_with(|| right.in_v11.cmp(&left.in_v11))
                .then_with(|| left.citation.cmp(&right.citation))
        });

        let mut target = python_round(bundle.estimated_count as f64 * cfg.target_mult);
        target = ((target as i32) + cfg.target_bias).max(cfg.min_output as i32) as usize;
        target = target.min(cfg.max_output);
        let court_cap = python_round(target as f64 * cfg.court_cap_frac);
        let mut law_base_counts: HashMap<String, usize> = HashMap::new();
        let mut court_base_counts: HashMap<String, usize> = HashMap::new();

        let mut selected = BTreeSet::new();
        let mut courts = 0usize;
        for item in ranked {
            if selected.len() >= target {
                break;
            }
            if item.kind == "court" && courts >= court_cap {
                continue;
            }
            if item.kind == "law" && cfg.max_law_per_base > 0 {
                if let Some(base) = law_base(&item.citation) {
                    if law_base_counts.get(&base).copied().unwrap_or(0) >= cfg.max_law_per_base {
                        continue;
                    }
                }
            }
            if item.kind == "court" && cfg.max_court_per_base > 0 {
                if let Some(base) = court_base(&item.citation) {
                    if court_base_counts.get(&base).copied().unwrap_or(0) >= cfg.max_court_per_base {
                        continue;
                    }
                }
            }
            if selected.insert(item.citation.clone()) && item.kind == "court" {
                courts += 1;
            }
            if item.kind == "law" {
                if let Some(base) = law_base(&item.citation) {
                    *law_base_counts.entry(base).or_insert(0) += 1;
                }
            } else if let Some(base) = court_base(&item.citation) {
                *court_base_counts.entry(base).or_insert(0) += 1;
            }
        }

        predictions.insert(bundle.query_id.clone(), selected.clone());
        if let Some(gold_citations) = gold_map.get(bundle.query_id.as_str()) {
            if has_gold {
                let score = f1(&selected, gold_citations);
                macro_f1 += score;
                macro_count += 1;
                per_query_f1.push(score);
            }
        }
    }

    let (avg_prediction_count, avg_court_fraction) = prediction_stats(&predictions);
    EvalResult {
        macro_f1: if has_gold && macro_count > 0 {
            Some(macro_f1 / macro_count as f64)
        } else {
            None
        },
        per_query_f1,
        predictions,
        avg_prediction_count,
        avg_court_fraction,
    }
}

fn robust_objective(
    val_result: &EvalResult,
    test_result: Option<&EvalResult>,
    test_v7_predictions: Option<&HashMap<String, BTreeSet<String>>>,
    test_v11_predictions: Option<&HashMap<String, BTreeSet<String>>>,
) -> f64 {
    let macro_f1 = val_result.macro_f1.unwrap_or(0.0);
    let query_std = stddev(&val_result.per_query_f1);
    let query_min = val_result
        .per_query_f1
        .iter()
        .copied()
        .fold(1.0, f64::min);

    let mut objective = macro_f1;
    objective -= 0.45 * query_std;
    objective += 0.08 * query_min;
    objective -= low_penalty(val_result.avg_court_fraction, 0.28, 0.9);
    objective -= low_penalty(val_result.avg_prediction_count, 24.0, 0.01);

    if let Some(test_result) = test_result {
        objective -= low_penalty(test_result.avg_court_fraction, 0.24, 0.9);
        objective -= low_penalty(test_result.avg_prediction_count, 23.0, 0.01);

        if let Some(v11) = test_v11_predictions {
            let jacc = jaccard_average(&test_result.predictions, v11);
            objective -= low_penalty(jacc, 0.60, 0.15);
        }
        if let Some(v7) = test_v7_predictions {
            let jacc = jaccard_average(&test_result.predictions, v7);
            objective -= low_penalty(jacc, 0.30, 0.10);
        }
    }

    objective
}

fn push_top_results(top_results: &mut Vec<CandidateResult>, candidate: CandidateResult, keep_top: usize) {
    top_results.push(candidate);
    top_results.sort_by(|left, right| cmp_desc_f64(left.objective, right.objective));
    if top_results.len() > keep_top {
        top_results.truncate(keep_top);
    }
}

fn consensus_predictions(
    top_results: &[CandidateResult],
    query_ids: &[String],
    use_test_predictions: bool,
    v7_predictions: &HashMap<String, BTreeSet<String>>,
    v11_predictions: &HashMap<String, BTreeSet<String>>,
) -> HashMap<String, BTreeSet<String>> {
    let mut consensus: HashMap<String, BTreeSet<String>> = HashMap::new();
    if top_results.is_empty() {
        return consensus;
    }

    for query_id in query_ids {
        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        let mut selected_sizes = Vec::new();
        let consensus_vote_frac: f64 = env::var("HYBRID_LAB_CONSENSUS_VOTE_FRAC")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.6);
        let consensus_target_bias: i32 = env::var("HYBRID_LAB_CONSENSUS_TARGET_BIAS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let consensus_target_mode = env::var("HYBRID_LAB_CONSENSUS_TARGET_MODE")
            .unwrap_or_else(|_| "median".to_string());
        for result in top_results {
            let source_predictions = if use_test_predictions {
                result
                    .test_result
                    .as_ref()
                    .map(|result| &result.predictions)
                    .unwrap_or(&result.val_result.predictions)
            } else {
                &result.val_result.predictions
            };
            if let Some(predictions) = source_predictions.get(query_id) {
                selected_sizes.push(predictions.len());
                for citation in predictions {
                    *vote_counts.entry(citation.clone()).or_insert(0) += 1;
                }
            }
        }
        selected_sizes.sort();
        if selected_sizes.is_empty() {
            consensus.insert(query_id.clone(), BTreeSet::new());
            continue;
        }
        let base_target = match consensus_target_mode.as_str() {
            "mean" => {
                let total = selected_sizes.iter().sum::<usize>() as f64;
                python_round(total / selected_sizes.len() as f64)
            }
            "max" => *selected_sizes.iter().max().unwrap_or(&selected_sizes[0]),
            _ => selected_sizes[selected_sizes.len() / 2],
        };
        let median_target = ((base_target as i32) + consensus_target_bias).max(1) as usize;
        let min_votes = ((top_results.len() as f64) * consensus_vote_frac).ceil() as usize;
        let v7 = v7_predictions.get(query_id).cloned().unwrap_or_default();
        let v11 = v11_predictions.get(query_id).cloned().unwrap_or_default();

        let mut ranked: Vec<(String, usize)> = vote_counts.into_iter().collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| v11.contains(&right.0).cmp(&v11.contains(&left.0)))
                .then_with(|| v7.contains(&right.0).cmp(&v7.contains(&left.0)))
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut selected = BTreeSet::new();
        for (citation, votes) in &ranked {
            if *votes >= min_votes {
                selected.insert(citation.clone());
            }
        }
        for (citation, _) in ranked {
            if selected.len() >= median_target {
                break;
            }
            selected.insert(citation);
        }
        consensus.insert(query_id.clone(), selected);
    }
    consensus
}

fn random_config(rng: &mut StdRng) -> HybridConfig {
    let v7_bonus = *[0.5_f64, 1.0_f64, 1.5_f64].choose(rng).unwrap();
    let v11_choices: &[f64] = if v7_bonus < 1.0 {
        &[1.0_f64, 1.5_f64, 2.0_f64]
    } else if v7_bonus < 1.5 {
        &[1.5_f64, 2.0_f64]
    } else {
        &[2.0_f64]
    };
    let v11_bonus = *v11_choices.choose(rng).unwrap();
    let both_bonus = *[0.0, 0.3, 0.6, 1.0, 1.5].choose(rng).unwrap();
    let auto_keep_bonus = *[0.0, 0.5, 1.0, 1.5].choose(rng).unwrap();
    let explicit_bonus = *[0.0, 0.5, 1.0, 2.0].choose(rng).unwrap();
    let must_bonus = *[0.5, 1.0, 1.5, 2.0, 3.0].choose(rng).unwrap();
    let plausible_bonus = *[0.0, 0.2, 0.5, 1.0, 1.5].choose(rng).unwrap();
    let reject_penalty = *[-2.0, -1.5, -1.0, -0.5, 0.0].choose(rng).unwrap();
    let conf_weight = *[0.0, 0.5, 1.0, 1.5, 2.0].choose(rng).unwrap();
    let final_weight = *[0.0, 0.5, 1.0, 1.5, 2.0].choose(rng).unwrap();
    let raw_weight = *[0.0, 0.3, 0.6, 1.0].choose(rng).unwrap();
    let gpt_freq_weight = *[0.0, 0.2, 0.5, 0.8].choose(rng).unwrap();
    let source_count_weight = *[-0.3, -0.1, 0.0, 0.1, 0.3].choose(rng).unwrap();
    let dense_bonus = *[-0.3, 0.0, 0.2, 0.5].choose(rng).unwrap();
    let bm25_bonus = *[-0.3, 0.0, 0.2, 0.5].choose(rng).unwrap();
    let gpt_case_bonus = *[-0.3, 0.0, 0.2, 0.5].choose(rng).unwrap();
    let cocitation_penalty = *[-1.0, -0.5, -0.2, 0.0, 0.2].choose(rng).unwrap();
    let court_dense_bonus = *[-0.8, -0.5, -0.2, 0.0, 0.2].choose(rng).unwrap();
    let law_bonus = *[-0.5, 0.0, 0.3, 0.6, 1.0].choose(rng).unwrap();
    let court_bonus = *[-1.0, -0.5, -0.2, 0.0].choose(rng).unwrap();
    let exact_train_weight = *[-0.3, -0.1, 0.0, 0.1, 0.3].choose(rng).unwrap();
    let dense_train_weight = *[0.0, 0.2, 0.5].choose(rng).unwrap();
    let law_base_train_weight = *[-0.3, -0.1, 0.0, 0.1, 0.3, 0.5].choose(rng).unwrap();
    let dense_law_base_train_weight = *[-0.5, -0.2, 0.0, 0.2, 0.5, 0.8].choose(rng).unwrap();
    let dense100_exact_weight = *[0.0, 0.1, 0.3, 0.5].choose(rng).unwrap();
    let dense12_exact_weight = *[0.0, 0.1, 0.3].choose(rng).unwrap();
    let sparse79_exact_weight = *[-0.5, -0.2, 0.0, 0.1].choose(rng).unwrap();
    let dense100_law_base_weight = *[0.0, 0.1, 0.3, 0.5].choose(rng).unwrap();
    let dense12_law_base_weight = *[0.0, 0.1, 0.3].choose(rng).unwrap();
    let sparse79_law_base_weight = *[-0.5, -0.2, 0.0, 0.1].choose(rng).unwrap();
    let court_dense_only_penalty = *[-1.5, -1.0, -0.5, 0.0].choose(rng).unwrap();
    let single_source_penalty = *[-0.8, -0.5, -0.2, 0.0].choose(rng).unwrap();
    let target_mult = *[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].choose(rng).unwrap();
    let target_bias = *[-10, -8, -6, -4, -2, 0, 2].choose(rng).unwrap();
    let min_output = *[6usize, 8, 10, 12].choose(rng).unwrap();
    let max_output = *[14usize, 16, 18, 20, 24, 28, 32].choose(rng).unwrap();
    let court_cap_frac = *[0.05, 0.1, 0.15, 0.2, 0.25, 0.3].choose(rng).unwrap();
    let max_law_per_base = *[0usize, 1, 2, 3, 4].choose(rng).unwrap();
    let max_court_per_base = *[1usize, 2, 3].choose(rng).unwrap();
    HybridConfig {
        v7_bonus,
        v11_bonus,
        both_bonus,
        auto_keep_bonus,
        explicit_bonus,
        must_bonus,
        plausible_bonus,
        reject_penalty,
        conf_weight,
        final_weight,
        raw_weight,
        gpt_freq_weight,
        source_count_weight,
        dense_bonus,
        bm25_bonus,
        gpt_case_bonus,
        cocitation_penalty,
        court_dense_bonus,
        law_bonus,
        court_bonus,
        exact_train_weight,
        dense_train_weight,
        law_base_train_weight,
        dense_law_base_train_weight,
        dense100_exact_weight,
        dense12_exact_weight,
        sparse79_exact_weight,
        dense100_law_base_weight,
        dense12_law_base_weight,
        sparse79_law_base_weight,
        court_dense_only_penalty,
        single_source_penalty,
        target_mult,
        target_bias,
        min_output,
        max_output: max_output.max(min_output),
        court_cap_frac,
        max_law_per_base,
        max_court_per_base,
    }
}

fn write_predictions(path: &PathBuf, predictions: &HashMap<String, BTreeSet<String>>) -> Result<(), Box<dyn Error>> {
    let mut writer = WriterBuilder::new().terminator(csv::Terminator::CRLF).from_path(path)?;
    writer.write_record(["query_id", "predicted_citations"])?;
    let mut keys: Vec<&String> = predictions.keys().collect();
    keys.sort();
    for key in keys {
        let citations = predictions.get(key).cloned().unwrap_or_default();
        writer.write_record([key.as_str(), &citations.into_iter().collect::<Vec<_>>().join(";")])?;
    }
    writer.flush()?;
    Ok(())
}

fn load_base_config(path: &PathBuf) -> Option<HybridConfig> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<HybridConfig>(&text).ok())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let judged_json = PathBuf::from(
        args.next()
            .ok_or("usage: hybrid_lab <judged_json> <v7_csv> <train_csv> <output_csv> [iterations] [best_cfg_json]")?,
    );
    let v7_csv = PathBuf::from(args.next().ok_or("missing v7 csv")?);
    let train_csv = PathBuf::from(args.next().ok_or("missing train csv")?);
    let output_csv = PathBuf::from(args.next().ok_or("missing output csv")?);
    let iterations: usize = args
        .next()
        .unwrap_or_else(|| "20000".to_string())
        .parse()
        .unwrap_or(20000);
    let best_cfg_json = args.next().map(PathBuf::from);
    let paired_judged_json = args.next().map(PathBuf::from);
    let paired_v7_csv = args.next().map(PathBuf::from);

    let artifact: Artifact = serde_json::from_str(&fs::read_to_string(&judged_json)?)?;
    let v7_predictions = parse_prediction_csv(&v7_csv)?;
    let priors = load_train_priors(&train_csv)?;

    let mut v11_predictions: HashMap<String, BTreeSet<String>> = HashMap::new();
    for bundle in &artifact.bundles {
        let selected = select_v11_bundle(bundle, &artifact.config);
        v11_predictions.insert(
            bundle.query_id.clone(),
            selected.iter().map(|candidate| candidate.citation.clone()).collect(),
        );
    }

    let mut best_score = f64::MIN;
    let mut best_cfg: Option<HybridConfig> = None;
    let mut best_predictions = HashMap::new();
    let mode = env::var("HYBRID_LAB_MODE").unwrap_or_else(|_| "random".to_string());
    let output_target = env::var("HYBRID_LAB_OUTPUT_TARGET").unwrap_or_else(|_| "val".to_string());
    let keep_top: usize = env::var("HYBRID_LAB_KEEP_TOP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);

    let paired_artifact = if let Some(path) = paired_judged_json.as_ref() {
        Some(serde_json::from_str::<Artifact>(&fs::read_to_string(path)?)?)
    } else {
        None
    };
    let paired_v7_predictions = if let Some(path) = paired_v7_csv.as_ref() {
        Some(parse_prediction_csv(path)?)
    } else {
        None
    };
    let mut paired_v11_predictions: Option<HashMap<String, BTreeSet<String>>> = None;
    if let Some(artifact) = paired_artifact.as_ref() {
        let mut predictions = HashMap::new();
        for bundle in &artifact.bundles {
            let selected = select_v11_bundle(bundle, &artifact.config);
            predictions.insert(
                bundle.query_id.clone(),
                selected.iter().map(|candidate| candidate.citation.clone()).collect(),
            );
        }
        paired_v11_predictions = Some(predictions);
    }
    let mut top_results: Vec<CandidateResult> = Vec::new();

    if mode == "apply" {
        let cfg_path = best_cfg_json.as_ref().ok_or("apply mode requires config path")?;
        let cfg = load_base_config(cfg_path).ok_or("failed to load config json")?;
        let result = evaluate_config(&artifact, &v7_predictions, &v11_predictions, &cfg, &priors);
        best_score = result.macro_f1.unwrap_or(0.0);
        best_cfg = Some(cfg);
        best_predictions = result.predictions;
    } else if mode == "grid" {
        let base_cfg = best_cfg_json
            .as_ref()
            .and_then(load_base_config)
            .unwrap_or(HybridConfig {
                v7_bonus: 0.5,
                v11_bonus: 2.0,
                both_bonus: 0.0,
                auto_keep_bonus: 0.5,
                explicit_bonus: 0.0,
                must_bonus: 1.0,
                plausible_bonus: 0.5,
                reject_penalty: 0.0,
                conf_weight: 1.5,
                final_weight: 2.0,
                raw_weight: 0.3,
                gpt_freq_weight: 0.8,
                source_count_weight: 0.0,
                dense_bonus: 0.0,
                bm25_bonus: 0.0,
                gpt_case_bonus: 0.0,
                cocitation_penalty: 0.0,
                court_dense_bonus: 0.0,
                law_bonus: 0.0,
                court_bonus: -0.5,
                exact_train_weight: 0.0,
                dense_train_weight: 0.0,
                law_base_train_weight: 0.0,
                dense_law_base_train_weight: 0.0,
                dense100_exact_weight: 0.0,
                dense12_exact_weight: 0.0,
                sparse79_exact_weight: 0.0,
                dense100_law_base_weight: 0.0,
                dense12_law_base_weight: 0.0,
                sparse79_law_base_weight: 0.0,
                court_dense_only_penalty: 0.0,
                single_source_penalty: -0.5,
                target_mult: 1.0,
                target_bias: 2,
                min_output: 8,
                max_output: 32,
                court_cap_frac: 0.2,
                max_law_per_base: 0,
                max_court_per_base: 0,
            });

        let both_bonus_values = [0.0, 0.5, 1.0];
        let court_bonus_values = [-1.0, -0.8, -0.5, -0.2, 0.0];
        let court_dense_bonus_values = [-0.8, -0.5, -0.2, 0.0];
        let exact_train_values = [0.0, 0.1, 0.3];
        let law_base_train_values = [0.0, 0.1, 0.3, 0.5];
        let dense_law_base_train_values = [0.0, 0.2, 0.5, 0.8];
        let max_law_per_base_values = [0usize, 1, 2, 3];
        let max_court_per_base_values = [0usize, 1, 2];
        let court_cap_values = [0.15, 0.2, 0.25];

        let total = both_bonus_values.len()
            * court_bonus_values.len()
            * court_dense_bonus_values.len()
            * exact_train_values.len()
            * law_base_train_values.len()
            * dense_law_base_train_values.len()
            * max_law_per_base_values.len()
            * max_court_per_base_values.len()
            * court_cap_values.len();
        println!("grid_total={total}");

        let mut idx = 0usize;
        for both_bonus in both_bonus_values {
            for court_bonus in court_bonus_values {
                for court_dense_bonus in court_dense_bonus_values {
                    for exact_train_weight in exact_train_values {
                        for law_base_train_weight in law_base_train_values {
                            for dense_law_base_train_weight in dense_law_base_train_values {
                                for max_law_per_base in max_law_per_base_values {
                                    for max_court_per_base in max_court_per_base_values {
                                        for court_cap_frac in court_cap_values {
                                            idx += 1;
                                            let mut cfg = base_cfg.clone();
                                            cfg.both_bonus = both_bonus;
                                            cfg.court_bonus = court_bonus;
                                            cfg.court_dense_bonus = court_dense_bonus;
                                            cfg.exact_train_weight = exact_train_weight;
                                            cfg.law_base_train_weight = law_base_train_weight;
                                            cfg.dense_law_base_train_weight = dense_law_base_train_weight;
                                            cfg.max_law_per_base = max_law_per_base;
                                            cfg.max_court_per_base = max_court_per_base;
                                            cfg.court_cap_frac = court_cap_frac;

                                            let result = evaluate_config(
                                                &artifact,
                                                &v7_predictions,
                                                &v11_predictions,
                                                &cfg,
                                                &priors,
                                            );
                                            let score = result.macro_f1.unwrap_or(0.0);
                                            if score > best_score {
                                                best_score = score;
                                                best_cfg = Some(cfg.clone());
                                                best_predictions = result.predictions;
                                                println!("iter {idx}/{total}: best {:.6} {:?}", best_score, cfg);
                                            } else if idx % 10000 == 0 {
                                                println!("iter {idx}/{total}: current_best {:.6}", best_score);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if mode == "robust" || mode == "consensus" {
        let mut rng = StdRng::seed_from_u64(0);
        for i in 0..iterations {
            let cfg = random_config(&mut rng);
            let val_result = evaluate_config(&artifact, &v7_predictions, &v11_predictions, &cfg, &priors);
            let test_result = if let (Some(paired_artifact), Some(paired_v7_predictions), Some(paired_v11_predictions)) = (
                paired_artifact.as_ref(),
                paired_v7_predictions.as_ref(),
                paired_v11_predictions.as_ref(),
            ) {
                Some(evaluate_config(
                    paired_artifact,
                    paired_v7_predictions,
                    paired_v11_predictions,
                    &cfg,
                    &priors,
                ))
            } else {
                None
            };
            let objective = robust_objective(
                &val_result,
                test_result.as_ref(),
                paired_v7_predictions.as_ref(),
                paired_v11_predictions.as_ref(),
            );
            let candidate = CandidateResult {
                objective,
                val_result,
                test_result,
            };
            push_top_results(&mut top_results, candidate.clone(), keep_top);
            if objective > best_score {
                best_score = objective;
                best_cfg = Some(cfg.clone());
                best_predictions = if output_target == "test" {
                    candidate
                        .test_result
                        .as_ref()
                        .map(|result| result.predictions.clone())
                        .unwrap_or_else(|| candidate.val_result.predictions.clone())
                } else {
                    candidate.val_result.predictions.clone()
                };
                let val_macro = candidate.val_result.macro_f1.unwrap_or(0.0);
                let test_count = candidate
                    .test_result
                    .as_ref()
                    .map(|result| result.avg_prediction_count)
                    .unwrap_or(0.0);
                let test_court = candidate
                    .test_result
                    .as_ref()
                    .map(|result| result.avg_court_fraction)
                    .unwrap_or(0.0);
                println!(
                    "iter {i}: objective {:.6} val {:.6} val_court {:.3} test_avg {:.2} test_court {:.3} {:?}",
                    best_score,
                    val_macro,
                    candidate.val_result.avg_court_fraction,
                    test_count,
                    test_court,
                    cfg
                );
            }
        }
        if mode == "consensus" && !top_results.is_empty() {
            let (query_ids, use_test, ref_v7, ref_v11) = if output_target == "test" {
                let artifact = paired_artifact
                    .as_ref()
                    .ok_or("consensus test output requires paired judged json")?;
                let ref_v7 = paired_v7_predictions
                    .as_ref()
                    .ok_or("consensus test output requires paired v7 csv")?;
                let ref_v11 = paired_v11_predictions
                    .as_ref()
                    .ok_or("consensus test output requires paired v11 predictions")?;
                (
                    artifact
                        .bundles
                        .iter()
                        .map(|bundle| bundle.query_id.clone())
                        .collect::<Vec<_>>(),
                    true,
                    ref_v7,
                    ref_v11,
                )
            } else {
                (
                    artifact
                        .bundles
                        .iter()
                        .map(|bundle| bundle.query_id.clone())
                        .collect::<Vec<_>>(),
                    false,
                    &v7_predictions,
                    &v11_predictions,
                )
            };
            best_predictions = consensus_predictions(&top_results, &query_ids, use_test, ref_v7, ref_v11);
        }
    } else {
        let mut rng = StdRng::seed_from_u64(0);
        for i in 0..iterations {
            let cfg = random_config(&mut rng);
            let result = evaluate_config(&artifact, &v7_predictions, &v11_predictions, &cfg, &priors);
            let score = result.macro_f1.unwrap_or(0.0);
            if score > best_score {
                best_score = score;
                best_cfg = Some(cfg.clone());
                best_predictions = result.predictions;
                println!("iter {i}: best {:.4} {:?}", best_score, cfg);
            }
        }
    }

    let best_cfg = best_cfg.ok_or("no config sampled")?;
    write_predictions(&output_csv, &best_predictions)?;
    let score_label = if mode == "robust" || mode == "consensus" {
        "best_objective"
    } else {
        "best_macro_f1"
    };
    println!("{score_label}={best_score:.6}");
    println!("best_config={}", serde_json::to_string_pretty(&best_cfg)?);

    if let Some(path) = best_cfg_json {
        fs::write(path, serde_json::to_string_pretty(&best_cfg)?)?;
    }

    Ok(())
}
