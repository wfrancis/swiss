use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::error::Error;

#[derive(Debug, Deserialize)]
struct Row {
    query_id: String,
    #[serde(default)]
    predicted_citations: String,
    #[serde(default)]
    gold_citations: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <mode:union|intersect> <gold.csv> <out.csv> <pred1.csv> <pred2.csv> [predN.csv...]",
            args[0]
        );
        std::process::exit(2);
    }
    let mode = &args[1];
    let gold = load_gold(&args[2])?;
    let out_path = &args[3];
    let preds = args[4..]
        .iter()
        .map(|path| load_pred(path))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = BTreeMap::new();
    for query_id in gold.keys() {
        let set = if mode == "intersect" {
            let mut iter = preds.iter();
            let mut acc = iter
                .next()
                .and_then(|p| p.get(query_id).cloned())
                .unwrap_or_default();
            for pred in iter {
                let row = pred.get(query_id).cloned().unwrap_or_default();
                acc = acc.intersection(&row).cloned().collect();
            }
            acc
        } else {
            let mut acc = BTreeSet::new();
            for pred in &preds {
                if let Some(row) = pred.get(query_id) {
                    acc.extend(row.iter().cloned());
                }
            }
            acc
        };
        out.insert(query_id.clone(), set);
    }
    write_pred(out_path, &out)?;
    eprintln!("macro_f1={:.6} mean_pred={:.1}", score(&gold, &out), mean_pred(&out));
    Ok(())
}

fn load_gold(path: &str) -> Result<BTreeMap<String, BTreeSet<String>>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().from_path(path)?;
    let mut out = BTreeMap::new();
    for row in reader.deserialize::<Row>() {
        let row = row?;
        out.insert(row.query_id, parse(&row.gold_citations));
    }
    Ok(out)
}

fn load_pred(path: &str) -> Result<BTreeMap<String, BTreeSet<String>>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().from_path(path)?;
    let mut out = BTreeMap::new();
    for row in reader.deserialize::<Row>() {
        let row = row?;
        out.insert(row.query_id, parse(&row.predicted_citations));
    }
    Ok(out)
}

fn write_pred(path: &str, rows: &BTreeMap<String, BTreeSet<String>>) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_path(path)?;
    writer.write_record(["query_id", "predicted_citations"])?;
    for (query_id, citations) in rows {
        writer.write_record([query_id.as_str(), &citations.iter().cloned().collect::<Vec<_>>().join(";")])?;
    }
    writer.flush()?;
    Ok(())
}

fn parse(raw: &str) -> BTreeSet<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn score(gold: &BTreeMap<String, BTreeSet<String>>, pred: &BTreeMap<String, BTreeSet<String>>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (query_id, gold_set) in gold {
        let pred_set = pred.get(query_id).cloned().unwrap_or_default();
        let tp = pred_set.intersection(gold_set).count() as f64;
        let f1 = if tp == 0.0 { 0.0 } else { 2.0 * tp / (pred_set.len() + gold_set.len()) as f64 };
        total += f1;
        count += 1;
    }
    total / count.max(1) as f64
}

fn mean_pred(pred: &BTreeMap<String, BTreeSet<String>>) -> f64 {
    pred.values().map(BTreeSet::len).sum::<usize>() as f64 / pred.len().max(1) as f64
}
