use csv::{ReaderBuilder, WriterBuilder};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::BufWriter;

type AnyError = Box<dyn Error + Send + Sync>;
type Citations = BTreeSet<String>;
type Predictions = BTreeMap<String, Citations>;

#[derive(Debug, Deserialize)]
struct CsvRow {
    query_id: String,
    #[serde(default)]
    query: String,
    #[serde(default)]
    predicted_citations: String,
    #[serde(default)]
    gold_citations: String,
}

#[derive(Debug, Clone)]
struct GoldRow {
    query_id: String,
    gold: Citations,
    bucket: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CandidateSpec {
    name: String,
    pred_path: String,
    #[serde(default)]
    test_path: String,
    #[serde(default)]
    public_score: String,
    #[serde(default)]
    note: String,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    name: String,
    pred_path: String,
    test_path: String,
    note: String,
    public_score: Option<f64>,
    coverage: usize,
    local_macro_f1: f64,
    query_std: f64,
    mean_public: f64,
    mean_private: f64,
    std_private: f64,
    p_public_winner: f64,
    p_private_winner: f64,
    p_public_winner_private_winner: f64,
    p_public_winner_private_top2: f64,
    p_public_winner_private_drop_gt_1pp: f64,
    p_public_winner_private_drop_gt_2pp: f64,
    mean_private_rank: f64,
    p_private_top2: f64,
    pessimistic_score: f64,
}

#[derive(Debug, Serialize)]
struct PairReport {
    left: String,
    right: String,
    mean_best_private: f64,
    std_best_private: f64,
    p_contains_private_winner: f64,
    p_both_private_top2: f64,
    p_one_private_top2: f64,
    test_jaccard: Option<f64>,
    local_prediction_jaccard: f64,
    public_score_max: Option<f64>,
    diversity_adjusted_score: f64,
}

#[derive(Debug)]
struct SplitOutcome {
    public_scores: Vec<f64>,
    private_scores: Vec<f64>,
    public_winner: usize,
    private_winner: usize,
    private_ranks: Vec<usize>,
}

#[derive(Debug, Clone)]
struct SimulationStats {
    split_count: usize,
    public_sum: Vec<f64>,
    private_sum: Vec<f64>,
    private_sum_sq: Vec<f64>,
    public_win: Vec<usize>,
    private_win: Vec<usize>,
    public_win_private_win: Vec<usize>,
    public_win_private_top2: Vec<usize>,
    public_win_drop_1pp: Vec<usize>,
    public_win_drop_2pp: Vec<usize>,
    private_rank_sum: Vec<usize>,
    private_top2: Vec<usize>,
    pair_best_sum: Vec<f64>,
    pair_best_sum_sq: Vec<f64>,
    pair_contains_private_winner: Vec<usize>,
    pair_both_private_top2: Vec<usize>,
    pair_one_private_top2: Vec<usize>,
    public_scores: Vec<f64>,
    private_scores: Vec<f64>,
    private_ranks: Vec<usize>,
    rank_order: Vec<usize>,
    public_idx: Vec<usize>,
    private_idx: Vec<usize>,
    split_scratch: Vec<usize>,
    key_scratch: Vec<String>,
}

impl SimulationStats {
    fn new(candidate_count: usize, row_capacity: usize, bucket_count: usize) -> Self {
        let pair_count = pair_count(candidate_count);
        SimulationStats {
            split_count: 0,
            public_sum: vec![0.0; candidate_count],
            private_sum: vec![0.0; candidate_count],
            private_sum_sq: vec![0.0; candidate_count],
            public_win: vec![0; candidate_count],
            private_win: vec![0; candidate_count],
            public_win_private_win: vec![0; candidate_count],
            public_win_private_top2: vec![0; candidate_count],
            public_win_drop_1pp: vec![0; candidate_count],
            public_win_drop_2pp: vec![0; candidate_count],
            private_rank_sum: vec![0; candidate_count],
            private_top2: vec![0; candidate_count],
            pair_best_sum: vec![0.0; pair_count],
            pair_best_sum_sq: vec![0.0; pair_count],
            pair_contains_private_winner: vec![0; pair_count],
            pair_both_private_top2: vec![0; pair_count],
            pair_one_private_top2: vec![0; pair_count],
            public_scores: vec![0.0; candidate_count],
            private_scores: vec![0.0; candidate_count],
            private_ranks: vec![0; candidate_count],
            rank_order: (0..candidate_count).collect(),
            public_idx: Vec::with_capacity(row_capacity),
            private_idx: Vec::with_capacity(row_capacity),
            split_scratch: Vec::with_capacity(row_capacity),
            key_scratch: Vec::with_capacity(bucket_count),
        }
    }

    fn observe(
        &mut self,
        scores: &[Vec<f64>],
        buckets: &BTreeMap<String, Vec<usize>>,
        public_frac: f64,
        scheme: SplitScheme,
        rng: &mut StdRng,
    ) {
        make_split_into(
            buckets,
            public_frac,
            scheme,
            rng,
            &mut self.public_idx,
            &mut self.private_idx,
            &mut self.split_scratch,
            &mut self.key_scratch,
        );
        for (idx, candidate_scores) in scores.iter().enumerate() {
            self.public_scores[idx] = mean_at(candidate_scores, &self.public_idx);
            self.private_scores[idx] = mean_at(candidate_scores, &self.private_idx);
        }

        let public_winner = argmax(&self.public_scores);
        let private_winner = argmax(&self.private_scores);
        ranks_desc_into(
            &self.private_scores,
            &mut self.rank_order,
            &mut self.private_ranks,
        );
        let best_private = self.private_scores[private_winner];

        self.split_count += 1;
        for idx in 0..scores.len() {
            let public_score = self.public_scores[idx];
            let private_score = self.private_scores[idx];
            let private_rank = self.private_ranks[idx];
            self.public_sum[idx] += public_score;
            self.private_sum[idx] += private_score;
            self.private_sum_sq[idx] += private_score * private_score;
            self.private_rank_sum[idx] += private_rank;
            if public_winner == idx {
                self.public_win[idx] += 1;
                if private_winner == idx {
                    self.public_win_private_win[idx] += 1;
                }
                if private_rank <= 2 {
                    self.public_win_private_top2[idx] += 1;
                }
                let private_drop = best_private - private_score;
                if private_drop > 0.01 {
                    self.public_win_drop_1pp[idx] += 1;
                }
                if private_drop > 0.02 {
                    self.public_win_drop_2pp[idx] += 1;
                }
            }
            if private_winner == idx {
                self.private_win[idx] += 1;
            }
            if private_rank <= 2 {
                self.private_top2[idx] += 1;
            }
        }

        let candidate_count = scores.len();
        for left in 0..candidate_count {
            let left_private = self.private_scores[left];
            let left_top2 = self.private_ranks[left] <= 2;
            for right in (left + 1)..candidate_count {
                let pair_idx = pair_index(candidate_count, left, right);
                let best_pair_private = left_private.max(self.private_scores[right]);
                self.pair_best_sum[pair_idx] += best_pair_private;
                self.pair_best_sum_sq[pair_idx] += best_pair_private * best_pair_private;
                let right_top2 = self.private_ranks[right] <= 2;
                if private_winner == left || private_winner == right {
                    self.pair_contains_private_winner[pair_idx] += 1;
                }
                if left_top2 && right_top2 {
                    self.pair_both_private_top2[pair_idx] += 1;
                }
                if left_top2 || right_top2 {
                    self.pair_one_private_top2[pair_idx] += 1;
                }
            }
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.split_count += other.split_count;
        add_f64(&mut self.public_sum, &other.public_sum);
        add_f64(&mut self.private_sum, &other.private_sum);
        add_f64(&mut self.private_sum_sq, &other.private_sum_sq);
        add_usize(&mut self.public_win, &other.public_win);
        add_usize(&mut self.private_win, &other.private_win);
        add_usize(
            &mut self.public_win_private_win,
            &other.public_win_private_win,
        );
        add_usize(
            &mut self.public_win_private_top2,
            &other.public_win_private_top2,
        );
        add_usize(&mut self.public_win_drop_1pp, &other.public_win_drop_1pp);
        add_usize(&mut self.public_win_drop_2pp, &other.public_win_drop_2pp);
        add_usize(&mut self.private_rank_sum, &other.private_rank_sum);
        add_usize(&mut self.private_top2, &other.private_top2);
        add_f64(&mut self.pair_best_sum, &other.pair_best_sum);
        add_f64(&mut self.pair_best_sum_sq, &other.pair_best_sum_sq);
        add_usize(
            &mut self.pair_contains_private_winner,
            &other.pair_contains_private_winner,
        );
        add_usize(
            &mut self.pair_both_private_top2,
            &other.pair_both_private_top2,
        );
        add_usize(
            &mut self.pair_one_private_top2,
            &other.pair_one_private_top2,
        );
        self
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SplitScheme {
    Combo,
    Domain,
    Proc,
    Length,
    GoldSize,
    Random,
    LeaveDomain,
    LeaveProc,
    LeaveLength,
    LeaveGoldSize,
}

impl Default for SplitScheme {
    fn default() -> Self {
        SplitScheme::Combo
    }
}

#[derive(Debug, Default)]
struct Cli {
    gold: String,
    candidates: String,
    out_dir: String,
    splits: usize,
    public_frac: f64,
    seed: u64,
    top_pairs: usize,
    scheme: SplitScheme,
}

fn main() -> Result<(), AnyError> {
    let cli = parse_args()?;
    fs::create_dir_all(&cli.out_dir)?;

    let gold_rows = load_gold(&cli.gold, cli.scheme)?;
    let qids = gold_rows
        .iter()
        .map(|row| row.query_id.clone())
        .collect::<Vec<_>>();
    let gold = gold_rows
        .iter()
        .map(|row| (row.query_id.clone(), row.gold.clone()))
        .collect::<BTreeMap<_, _>>();
    let specs = load_specs(&cli.candidates)?;
    if specs.len() < 2 {
        return Err("need at least two candidates".into());
    }
    let mut local_preds = Vec::new();
    let mut test_preds = Vec::new();
    let mut scores = Vec::new();
    let mut coverage = Vec::new();
    for spec in &specs {
        let pred = load_predictions(&spec.pred_path)?;
        coverage.push(qids.iter().filter(|qid| pred.contains_key(*qid)).count());
        scores.push(
            qids.iter()
                .map(|qid| f1(&pred.get(qid).cloned().unwrap_or_default(), &gold[qid]))
                .collect::<Vec<_>>(),
        );
        local_preds.push(pred);
        if spec.test_path.trim().is_empty() {
            test_preds.push(None);
        } else {
            test_preds.push(Some(load_predictions(&spec.test_path)?));
        }
    }

    let buckets = bucket_indices(&gold_rows);
    eprintln!(
        "loaded rows={} buckets={} candidates={} splits={} public_frac={:.3} scheme={}",
        gold_rows.len(),
        buckets.len(),
        specs.len(),
        cli.splits,
        cli.public_frac,
        scheme_name(cli.scheme)
    );

    let stats = (0..cli.splits)
        .into_par_iter()
        .fold(
            || SimulationStats::new(specs.len(), gold_rows.len(), buckets.len()),
            |mut stats, split_idx| {
                let mut rng = StdRng::seed_from_u64(cli.seed ^ hash_u64(split_idx));
                stats.observe(&scores, &buckets, cli.public_frac, cli.scheme, &mut rng);
                stats
            },
        )
        .reduce(
            || SimulationStats::new(specs.len(), gold_rows.len(), buckets.len()),
            SimulationStats::merge,
        );

    let candidate_reports = candidate_reports(
        &specs,
        &qids,
        &scores,
        &coverage,
        &stats,
        &local_preds,
    );
    let pair_reports = pair_reports(
        &specs,
        &qids,
        &stats,
        &local_preds,
        &test_preds,
        cli.top_pairs,
    );

    write_candidate_tsv(
        &format!("{}/candidate_report.tsv", cli.out_dir),
        &candidate_reports,
    )?;
    write_pair_tsv(&format!("{}/pair_report.tsv", cli.out_dir), &pair_reports)?;
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(format!("{}/summary.json", cli.out_dir))?),
        &serde_json::json!({
            "gold": cli.gold,
            "candidates": cli.candidates,
            "rows": gold_rows.len(),
            "buckets": buckets.keys().collect::<Vec<_>>(),
            "splits": cli.splits,
            "public_frac": cli.public_frac,
            "scheme": scheme_name(cli.scheme),
            "seed": cli.seed,
            "candidate_report": candidate_reports,
            "pair_report": pair_reports,
        }),
    )?;

    println!("PRIVATE SPLIT PORTFOLIO");
    println!(
        "rows={} buckets={} splits={} scheme={}",
        gold_rows.len(),
        buckets.len(),
        cli.splits,
        scheme_name(cli.scheme)
    );
    println!();
    println!("Top candidates by pessimistic private score:");
    for report in candidate_reports.iter().take(12) {
        println!(
            "{:<34} local={:.5} priv={:.5}±{:.5} p_priv_win={:.3} p_top2={:.3} p_pubwin_drop2={:.3} pess={:.5}",
            truncate(&report.name, 34),
            report.local_macro_f1,
            report.mean_private,
            report.std_private,
            report.p_private_winner,
            report.p_private_top2,
            report.p_public_winner_private_drop_gt_2pp,
            report.pessimistic_score
        );
    }
    println!();
    println!("Top final-submission pairs:");
    for pair in pair_reports.iter().take(cli.top_pairs.min(12)) {
        println!(
            "{:<28} + {:<28} best_priv={:.5}±{:.5} p_contains_win={:.3} test_j={}",
            truncate(&pair.left, 28),
            truncate(&pair.right, 28),
            pair.mean_best_private,
            pair.std_best_private,
            pair.p_contains_private_winner,
            pair.test_jaccard
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string())
        );
    }
    Ok(())
}

fn parse_args() -> Result<Cli, AnyError> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 1 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        eprintln!(
            "Usage: private_split_portfolio --gold data/train.csv --candidates manifest.tsv --out-dir artifacts/private_split --splits 20000 [--public-frac 0.5] [--scheme combo|domain|proc|length|gold|random|leave-domain|leave-proc|leave-length|leave-gold] [--seed 20260507] [--top-pairs 25]"
        );
        std::process::exit(2);
    }
    let mut cli = Cli {
        splits: 20_000,
        public_frac: 0.5,
        seed: 20260507,
        top_pairs: 25,
        scheme: SplitScheme::Combo,
        ..Cli::default()
    };
    let mut i = 1usize;
    while i < args.len() {
        let key = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?
            .clone();
        match key {
            "--gold" => cli.gold = value,
            "--candidates" => cli.candidates = value,
            "--out-dir" => cli.out_dir = value,
            "--splits" => cli.splits = value.parse()?,
            "--public-frac" => cli.public_frac = value.parse()?,
            "--scheme" => cli.scheme = parse_scheme(&value)?,
            "--seed" => cli.seed = value.parse()?,
            "--top-pairs" => cli.top_pairs = value.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 2;
    }
    if cli.gold.is_empty() || cli.candidates.is_empty() || cli.out_dir.is_empty() {
        return Err("missing required argument".into());
    }
    if !(0.05..=0.95).contains(&cli.public_frac) {
        return Err("--public-frac must be in [0.05, 0.95]".into());
    }
    if cli.splits == 0 {
        return Err("--splits must be positive".into());
    }
    Ok(cli)
}

fn parse_scheme(raw: &str) -> Result<SplitScheme, AnyError> {
    match raw {
        "combo" => Ok(SplitScheme::Combo),
        "domain" => Ok(SplitScheme::Domain),
        "proc" | "procedure" => Ok(SplitScheme::Proc),
        "length" | "len" => Ok(SplitScheme::Length),
        "gold" | "gold-size" | "gold_size" => Ok(SplitScheme::GoldSize),
        "random" => Ok(SplitScheme::Random),
        "leave-domain" | "leave_domain" => Ok(SplitScheme::LeaveDomain),
        "leave-proc" | "leave-procedure" | "leave_proc" => Ok(SplitScheme::LeaveProc),
        "leave-length" | "leave-len" | "leave_length" => Ok(SplitScheme::LeaveLength),
        "leave-gold" | "leave-gold-size" | "leave_gold" => Ok(SplitScheme::LeaveGoldSize),
        other => Err(format!("unknown split scheme: {other}").into()),
    }
}

fn scheme_name(scheme: SplitScheme) -> &'static str {
    match scheme {
        SplitScheme::Combo => "combo",
        SplitScheme::Domain => "domain",
        SplitScheme::Proc => "proc",
        SplitScheme::Length => "length",
        SplitScheme::GoldSize => "gold",
        SplitScheme::Random => "random",
        SplitScheme::LeaveDomain => "leave-domain",
        SplitScheme::LeaveProc => "leave-proc",
        SplitScheme::LeaveLength => "leave-length",
        SplitScheme::LeaveGoldSize => "leave-gold",
    }
}

fn scheme_is_leave_bucket(scheme: SplitScheme) -> bool {
    matches!(
        scheme,
        SplitScheme::LeaveDomain
            | SplitScheme::LeaveProc
            | SplitScheme::LeaveLength
            | SplitScheme::LeaveGoldSize
    )
}

fn load_gold(path: &str, scheme: SplitScheme) -> Result<Vec<GoldRow>, AnyError> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut rows = Vec::new();
    for row in reader.deserialize::<CsvRow>() {
        let row = row?;
        let gold = parse_set(&row.gold_citations);
        rows.push(GoldRow {
            query_id: row.query_id,
            bucket: make_bucket(&row.query, &gold, scheme),
            gold,
        });
    }
    Ok(rows)
}

fn load_specs(path: &str) -> Result<Vec<CandidateSpec>, AnyError> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let mut specs = Vec::new();
    for row in reader.deserialize::<CandidateSpec>() {
        let spec = row?;
        if spec.name.trim().is_empty() || spec.pred_path.trim().is_empty() {
            continue;
        }
        specs.push(spec);
    }
    Ok(specs)
}

fn load_predictions(path: &str) -> Result<Predictions, AnyError> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut out = Predictions::new();
    for row in reader.deserialize::<CsvRow>() {
        let row = row?;
        out.insert(row.query_id, parse_set(&row.predicted_citations));
    }
    Ok(out)
}

fn parse_set(raw: &str) -> Citations {
    raw.split(';')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn make_bucket(query: &str, gold: &Citations, scheme: SplitScheme) -> String {
    let domain = dominant_domain(gold);
    let proc = procedure_bucket(query, &domain);
    let len = length_bucket(query);
    let size = gold_size_bucket(gold);
    match scheme {
        SplitScheme::Combo => format!("{proc}|dom_{domain}|{len}|{size}"),
        SplitScheme::Domain | SplitScheme::LeaveDomain => format!("dom_{domain}"),
        SplitScheme::Proc | SplitScheme::LeaveProc => proc,
        SplitScheme::Length | SplitScheme::LeaveLength => len,
        SplitScheme::GoldSize | SplitScheme::LeaveGoldSize => size,
        SplitScheme::Random => "all".to_string(),
    }
}

fn procedure_bucket(query: &str, domain: &str) -> String {
    if query.contains("Untersuchungshaft")
        || query.contains("Strafverfahren")
        || domain == "stpo"
        || domain == "stgb"
    {
        "proc_criminal".to_string()
    } else if domain == "atsg" || domain == "ivg" || domain == "uvg" || query.contains("Sozial") {
        "proc_social".to_string()
    } else {
        "proc_civil".to_string()
    }
}

fn length_bucket(query: &str) -> String {
    let words = query.split_whitespace().count();
    match words {
        0..=100 => "len_short",
        101..=240 => "len_medium",
        _ => "len_long",
    }
    .to_string()
}

fn gold_size_bucket(gold: &Citations) -> String {
    match gold.len() {
        0..=5 => "gold_small",
        6..=12 => "gold_medium",
        _ => "gold_large",
    }
    .to_string()
}

fn dominant_domain(gold: &Citations) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for citation in gold {
        if citation.starts_with("Art.") {
            if let Some(code) = citation.split_whitespace().last() {
                *counts.entry(code.to_ascii_lowercase()).or_default() += 1;
            }
        } else if citation.starts_with("BGE") {
            *counts.entry("bge".to_string()).or_default() += 1;
        } else {
            *counts.entry("court".to_string()).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(domain, _)| domain)
        .unwrap_or_else(|| "none".to_string())
}

fn bucket_indices(rows: &[GoldRow]) -> BTreeMap<String, Vec<usize>> {
    let mut out = BTreeMap::<String, Vec<usize>>::new();
    for (idx, row) in rows.iter().enumerate() {
        out.entry(row.bucket.clone()).or_default().push(idx);
    }
    out
}

fn make_split_into(
    buckets: &BTreeMap<String, Vec<usize>>,
    public_frac: f64,
    scheme: SplitScheme,
    rng: &mut StdRng,
    public_idx: &mut Vec<usize>,
    private_idx: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
    key_scratch: &mut Vec<String>,
) {
    public_idx.clear();
    private_idx.clear();
    if scheme_is_leave_bucket(scheme) {
        make_leave_bucket_split_into(
            buckets,
            public_frac,
            rng,
            public_idx,
            private_idx,
            scratch,
            key_scratch,
        );
        return;
    }
    for indices in buckets.values() {
        scratch.clear();
        scratch.extend(indices.iter().copied());
        scratch.shuffle(rng);
        let public_count = if scratch.len() == 1 {
            if rng.gen_bool(public_frac) {
                1
            } else {
                0
            }
        } else {
            ((scratch.len() as f64 * public_frac).round() as usize)
                .max(1)
                .min(scratch.len() - 1)
        };
        for (rank, idx) in scratch.iter().copied().enumerate() {
            if rank < public_count {
                public_idx.push(idx);
            } else {
                private_idx.push(idx);
            }
        }
    }
}

fn make_leave_bucket_split_into(
    buckets: &BTreeMap<String, Vec<usize>>,
    public_frac: f64,
    rng: &mut StdRng,
    public_idx: &mut Vec<usize>,
    private_idx: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
    key_scratch: &mut Vec<String>,
) {
    let total_rows = buckets.values().map(Vec::len).sum::<usize>();
    let private_target = ((total_rows as f64 * (1.0 - public_frac)).round() as usize)
        .max(1)
        .min(total_rows.saturating_sub(1).max(1));
    key_scratch.clear();
    key_scratch.extend(buckets.keys().cloned());
    key_scratch.shuffle(rng);

    let mut private_count = 0usize;
    for key in key_scratch.iter() {
        let Some(indices) = buckets.get(key) else {
            continue;
        };
        if private_count < private_target {
            private_idx.extend(indices.iter().copied());
            private_count += indices.len();
        } else {
            public_idx.extend(indices.iter().copied());
        }
    }
    if public_idx.is_empty() || private_idx.is_empty() {
        scratch.clear();
        scratch.extend(buckets.values().flat_map(|indices| indices.iter().copied()));
        scratch.shuffle(rng);
        let public_count = ((scratch.len() as f64 * public_frac).round() as usize)
            .max(1)
            .min(scratch.len().saturating_sub(1).max(1));
        public_idx.clear();
        private_idx.clear();
        for (rank, idx) in scratch.iter().copied().enumerate() {
            if rank < public_count {
                public_idx.push(idx);
            } else {
                private_idx.push(idx);
            }
        }
    }
}

fn make_split(
    buckets: &BTreeMap<String, Vec<usize>>,
    public_frac: f64,
    scheme: SplitScheme,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>) {
    if scheme_is_leave_bucket(scheme) {
        return make_leave_bucket_split(buckets, public_frac, rng);
    }
    let mut public_idx = Vec::new();
    let mut private_idx = Vec::new();
    for indices in buckets.values() {
        let mut shuffled = indices.clone();
        shuffled.shuffle(rng);
        let public_count = if shuffled.len() == 1 {
            if rng.gen_bool(public_frac) {
                1
            } else {
                0
            }
        } else {
            ((shuffled.len() as f64 * public_frac).round() as usize)
                .max(1)
                .min(shuffled.len() - 1)
        };
        for (rank, idx) in shuffled.into_iter().enumerate() {
            if rank < public_count {
                public_idx.push(idx);
            } else {
                private_idx.push(idx);
            }
        }
    }
    (public_idx, private_idx)
}

fn make_leave_bucket_split(
    buckets: &BTreeMap<String, Vec<usize>>,
    public_frac: f64,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>) {
    let total_rows = buckets.values().map(Vec::len).sum::<usize>();
    let private_target = ((total_rows as f64 * (1.0 - public_frac)).round() as usize)
        .max(1)
        .min(total_rows.saturating_sub(1).max(1));
    let mut keys = buckets.keys().cloned().collect::<Vec<_>>();
    keys.shuffle(rng);

    let mut public_idx = Vec::new();
    let mut private_idx = Vec::new();
    let mut private_count = 0usize;
    for key in keys {
        let Some(indices) = buckets.get(&key) else {
            continue;
        };
        if private_count < private_target {
            private_idx.extend(indices.iter().copied());
            private_count += indices.len();
        } else {
            public_idx.extend(indices.iter().copied());
        }
    }
    if public_idx.is_empty() || private_idx.is_empty() {
        let all = buckets
            .values()
            .flat_map(|indices| indices.iter().copied())
            .collect::<Vec<_>>();
        let mut pseudo = BTreeMap::new();
        pseudo.insert("all".to_string(), all);
        return make_split(&pseudo, public_frac, SplitScheme::Random, rng);
    }
    (public_idx, private_idx)
}

fn split_outcome(
    scores: &[Vec<f64>],
    public_idx: &[usize],
    private_idx: &[usize],
) -> SplitOutcome {
    let public_scores = scores
        .iter()
        .map(|candidate| mean_at(candidate, public_idx))
        .collect::<Vec<_>>();
    let private_scores = scores
        .iter()
        .map(|candidate| mean_at(candidate, private_idx))
        .collect::<Vec<_>>();
    let public_winner = argmax(&public_scores);
    let private_winner = argmax(&private_scores);
    let private_ranks = ranks_desc(&private_scores);
    SplitOutcome {
        public_scores,
        private_scores,
        public_winner,
        private_winner,
        private_ranks,
    }
}

fn candidate_reports(
    specs: &[CandidateSpec],
    qids: &[String],
    scores: &[Vec<f64>],
    coverage: &[usize],
    stats: &SimulationStats,
    local_preds: &[Predictions],
) -> Vec<CandidateReport> {
    let split_n = stats.split_count.max(1) as f64;
    let candidate_count = specs.len();
    let mut reports = Vec::new();
    for idx in 0..candidate_count {
        let local_macro = mean(&scores[idx]);
        let mean_public = stats.public_sum[idx] / split_n;
        let mean_private = stats.private_sum[idx] / split_n;
        let std_private = variance_from_sums(
            stats.private_sum[idx],
            stats.private_sum_sq[idx],
            stats.split_count,
        )
        .sqrt();
        let p_public_winner = stats.public_win[idx] as f64 / split_n;
        let p_private_winner = stats.private_win[idx] as f64 / split_n;
        let p_public_winner_private_winner =
            stats.public_win_private_win[idx] as f64 / split_n;
        let p_public_winner_private_top2 =
            stats.public_win_private_top2[idx] as f64 / split_n;
        let public_wins = stats.public_win[idx].max(1) as f64;
        let p_drop_1 = stats.public_win_drop_1pp[idx] as f64 / public_wins;
        let p_drop_2 = stats.public_win_drop_2pp[idx] as f64 / public_wins;
        let mean_private_rank = stats.private_rank_sum[idx] as f64 / split_n;
        let p_private_top2 = stats.private_top2[idx] as f64 / split_n;
        let pessimistic_score = mean_private
            - 0.65 * std_private
            - 0.006 * p_drop_2
            - 0.002 * (1.0 - p_private_top2);
        reports.push(CandidateReport {
            name: specs[idx].name.clone(),
            pred_path: specs[idx].pred_path.clone(),
            test_path: specs[idx].test_path.clone(),
            note: specs[idx].note.clone(),
            public_score: parse_optional_f64(&specs[idx].public_score),
            coverage: coverage[idx],
            local_macro_f1: local_macro,
            query_std: std_pop(&scores[idx]),
            mean_public,
            mean_private,
            std_private,
            p_public_winner,
            p_private_winner,
            p_public_winner_private_winner,
            p_public_winner_private_top2,
            p_public_winner_private_drop_gt_1pp: p_drop_1,
            p_public_winner_private_drop_gt_2pp: p_drop_2,
            mean_private_rank,
            p_private_top2,
            pessimistic_score,
        });
    }
    reports.sort_by(|left, right| {
        cmp_f64(right.pessimistic_score, left.pessimistic_score)
            .then_with(|| cmp_f64(right.mean_private, left.mean_private))
            .then_with(|| left.name.cmp(&right.name))
    });
    let _ = qids;
    let _ = local_preds;
    reports
}

fn public_winner_drop_prob(outcomes: &[SplitOutcome], idx: usize, threshold: f64) -> f64 {
    let mut wins = 0usize;
    let mut drops = 0usize;
    for outcome in outcomes {
        if outcome.public_winner != idx {
            continue;
        }
        wins += 1;
        let best_private = outcome.private_scores[outcome.private_winner];
        if best_private - outcome.private_scores[idx] > threshold {
            drops += 1;
        }
    }
    if wins == 0 {
        0.0
    } else {
        drops as f64 / wins as f64
    }
}

fn pair_reports(
    specs: &[CandidateSpec],
    qids: &[String],
    stats: &SimulationStats,
    local_preds: &[Predictions],
    test_preds: &[Option<Predictions>],
    top_pairs: usize,
) -> Vec<PairReport> {
    let split_n = stats.split_count.max(1) as f64;
    let mut reports = Vec::new();
    for left in 0..specs.len() {
        for right in (left + 1)..specs.len() {
            let pair_idx = pair_index(specs.len(), left, right);
            let p_contains_private_winner =
                stats.pair_contains_private_winner[pair_idx] as f64 / split_n;
            let p_both_private_top2 =
                stats.pair_both_private_top2[pair_idx] as f64 / split_n;
            let p_one_private_top2 =
                stats.pair_one_private_top2[pair_idx] as f64 / split_n;
            let mean_best_private = stats.pair_best_sum[pair_idx] / split_n;
            let std_best_private = variance_from_sums(
                stats.pair_best_sum[pair_idx],
                stats.pair_best_sum_sq[pair_idx],
                stats.split_count,
            )
            .sqrt();
            let local_prediction_jaccard = prediction_jaccard_for_qids(&local_preds[left], &local_preds[right], qids);
            let test_jaccard = match (&test_preds[left], &test_preds[right]) {
                (Some(a), Some(b)) => Some(prediction_jaccard_all(a, b)),
                _ => None,
            };
            let public_score_max = match (
                parse_optional_f64(&specs[left].public_score),
                parse_optional_f64(&specs[right].public_score),
            ) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let diversity = test_jaccard.unwrap_or(local_prediction_jaccard);
            let diversity_penalty = (diversity - 0.985).max(0.0) * 0.006;
            let diversity_bonus = (0.965 - diversity).max(0.0).min(0.06) * 0.002;
            reports.push(PairReport {
                left: specs[left].name.clone(),
                right: specs[right].name.clone(),
                mean_best_private,
                std_best_private,
                p_contains_private_winner,
                p_both_private_top2,
                p_one_private_top2,
                test_jaccard,
                local_prediction_jaccard,
                public_score_max,
                diversity_adjusted_score: mean_best_private
                    - 0.45 * std_best_private
                    + 0.004 * p_contains_private_winner
                    + diversity_bonus
                    - diversity_penalty,
            });
        }
    }
    reports.sort_by(|left, right| {
        cmp_f64(right.diversity_adjusted_score, left.diversity_adjusted_score)
            .then_with(|| cmp_f64(right.mean_best_private, left.mean_best_private))
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    reports.truncate(top_pairs.max(1));
    reports
}

fn write_candidate_tsv(path: &str, rows: &[CandidateReport]) -> Result<(), AnyError> {
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pair_tsv(path: &str, rows: &[PairReport]) -> Result<(), AnyError> {
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn f1(pred: &Citations, gold: &Citations) -> f64 {
    if pred.is_empty() && gold.is_empty() {
        return 1.0;
    }
    let tp = pred.intersection(gold).count() as f64;
    if tp == 0.0 {
        0.0
    } else {
        2.0 * tp / (pred.len() + gold.len()) as f64
    }
}

fn mean_at(values: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|idx| values[*idx]).sum::<f64>() / indices.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_pop(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let avg = mean(values);
    (values
        .iter()
        .map(|value| {
            let delta = value - avg;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn variance_from_sums(sum: f64, sum_sq: f64, count: usize) -> f64 {
    if count <= 1 {
        return 0.0;
    }
    let n = count as f64;
    let mean = sum / n;
    (sum_sq / n - mean * mean).max(0.0)
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| cmp_f64(*left.1, *right.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn ranks_desc(values: &[f64]) -> Vec<usize> {
    let mut order = values.iter().copied().enumerate().collect::<Vec<_>>();
    order.sort_by(|left, right| cmp_f64(right.1, left.1).then_with(|| left.0.cmp(&right.0)));
    let mut ranks = vec![0usize; values.len()];
    for (rank, (idx, _)) in order.into_iter().enumerate() {
        ranks[idx] = rank + 1;
    }
    ranks
}

fn ranks_desc_into(values: &[f64], order: &mut [usize], ranks: &mut [usize]) {
    order.sort_unstable_by(|left, right| {
        cmp_f64(values[*right], values[*left]).then_with(|| left.cmp(right))
    });
    for (rank, idx) in order.iter().copied().enumerate() {
        ranks[idx] = rank + 1;
    }
}

fn pair_count(candidate_count: usize) -> usize {
    candidate_count.saturating_mul(candidate_count.saturating_sub(1)) / 2
}

fn pair_index(candidate_count: usize, left: usize, right: usize) -> usize {
    debug_assert!(left < right);
    left * (2 * candidate_count - left - 1) / 2 + (right - left - 1)
}

fn add_f64(left: &mut [f64], right: &[f64]) {
    for (left_value, right_value) in left.iter_mut().zip(right.iter()) {
        *left_value += right_value;
    }
}

fn add_usize(left: &mut [usize], right: &[usize]) {
    for (left_value, right_value) in left.iter_mut().zip(right.iter()) {
        *left_value += right_value;
    }
}

fn prediction_jaccard_for_qids(left: &Predictions, right: &Predictions, qids: &[String]) -> f64 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for qid in qids {
        let a = left.get(qid).cloned().unwrap_or_default();
        let b = right.get(qid).cloned().unwrap_or_default();
        intersection += a.intersection(&b).count();
        union += a.union(&b).count();
    }
    intersection as f64 / union.max(1) as f64
}

fn prediction_jaccard_all(left: &Predictions, right: &Predictions) -> f64 {
    let qids = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    prediction_jaccard_for_qids(left, right, &qids)
}

fn parse_optional_f64(raw: &str) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "-" {
        None
    } else {
        trimmed.parse::<f64>().ok()
    }
}

fn hash_u64<T: Hash>(value: T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn truncate(value: &str, max_len: usize) -> String {
    if value.chars().count() <= max_len {
        value.to_string()
    } else {
        let mut out = value.chars().take(max_len.saturating_sub(1)).collect::<String>();
        out.push('~');
        out
    }
}

fn cmp_f64(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}
