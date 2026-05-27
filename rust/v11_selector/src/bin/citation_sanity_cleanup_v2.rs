use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Deserialize)]
struct PredRow {
    query_id: String,
    #[serde(default)]
    predicted_citations: String,
}

#[derive(Debug, Deserialize)]
struct CitationRow {
    citation: String,
}

#[derive(Debug, Deserialize)]
struct GoldRow {
    #[serde(default)]
    gold_citations: String,
}

#[derive(Debug, Serialize)]
struct Removal {
    split: String,
    query_id: String,
    citation: String,
    reason: String,
}

#[derive(Debug, Serialize)]
struct SplitSummary {
    input: String,
    output: String,
    rows: usize,
    total_before: usize,
    total_after: usize,
    removed: usize,
}

#[derive(Debug, Serialize)]
struct Summary {
    rules: Vec<String>,
    law_universe_size: usize,
    court_universe_size: usize,
    gold_universe_size: usize,
    val: SplitSummary,
    test: SplitSummary,
    removals: Vec<Removal>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 10 {
        eprintln!(
            "Usage: {} <base_val.csv> <base_test.csv> <laws_de.csv> <court_considerations.csv> <train.csv> <val.csv> <out_val.csv> <out_test.csv> <summary.json>",
            args.first()
                .map(String::as_str)
                .unwrap_or("citation_sanity_cleanup_v2")
        );
        std::process::exit(2);
    }

    let law_universe = load_citation_set(&args[3])?;
    let court_universe = load_citation_set(&args[4])?;
    let mut gold_universe = load_gold_set(&args[5])?;
    gold_universe.extend(load_gold_set(&args[6])?);

    let bad_section_year = Regex::new(r"\sE\.\s+(?:19|20)\d{2}$")?;
    let mut removals = Vec::new();

    let val = clean_split(
        "val",
        &args[1],
        &args[7],
        &bad_section_year,
        &law_universe,
        &court_universe,
        &gold_universe,
        &mut removals,
    )?;
    let test = clean_split(
        "test",
        &args[2],
        &args[8],
        &bad_section_year,
        &law_universe,
        &court_universe,
        &gold_universe,
        &mut removals,
    )?;

    let summary = Summary {
        rules: vec![
            "remove citations whose terminal consideration section is a calendar year: / E. (19|20)\\d{2}$/".to_string(),
            "remove Art.* citations absent from laws_de, court_considerations, train gold, and val gold".to_string(),
        ],
        law_universe_size: law_universe.len(),
        court_universe_size: court_universe.len(),
        gold_universe_size: gold_universe.len(),
        val,
        test,
        removals,
    };
    let writer = BufWriter::new(File::create(&args[9])?);
    serde_json::to_writer_pretty(writer, &summary)?;
    eprintln!(
        "removed {} citation sanity failures",
        summary.removals.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn clean_split(
    split: &str,
    input_path: &str,
    output_path: &str,
    bad_section_year: &Regex,
    law_universe: &HashSet<String>,
    court_universe: &HashSet<String>,
    gold_universe: &HashSet<String>,
    removals: &mut Vec<Removal>,
) -> Result<SplitSummary, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(input_path)?;
    let mut wtr = csv::WriterBuilder::new().from_path(output_path)?;
    wtr.write_record(["query_id", "predicted_citations"])?;

    let mut rows = 0usize;
    let mut total_before = 0usize;
    let mut total_after = 0usize;
    let mut removed = 0usize;

    for result in rdr.deserialize() {
        let row: PredRow = result?;
        rows += 1;
        let before = parse_citations(&row.predicted_citations);
        total_before += before.len();

        let mut after = Vec::with_capacity(before.len());
        let mut row_removals = Vec::new();
        for citation in before.iter() {
            if bad_section_year.is_match(citation) {
                row_removals.push((
                    citation.clone(),
                    "terminal consideration section is a calendar year".to_string(),
                ));
            } else if citation.starts_with("Art. ")
                && !law_universe.contains(citation)
                && !court_universe.contains(citation)
                && !gold_universe.contains(citation)
            {
                row_removals.push((
                    citation.clone(),
                    "article citation absent from exact citation universe".to_string(),
                ));
            } else {
                after.push(citation.clone());
            }
        }

        // Guard against invalid empty submission rows.
        if after.is_empty() && !before.is_empty() {
            after = before.clone();
        } else {
            removed += row_removals.len();
            for (citation, reason) in row_removals {
                removals.push(Removal {
                    split: split.to_string(),
                    query_id: row.query_id.clone(),
                    citation,
                    reason,
                });
            }
        }

        total_after += after.len();
        wtr.write_record([row.query_id.as_str(), &after.join(";")])?;
    }
    wtr.flush()?;

    Ok(SplitSummary {
        input: input_path.to_string(),
        output: output_path.to_string(),
        rows,
        total_before,
        total_after,
        removed,
    })
}

fn load_citation_set(path: &str) -> Result<HashSet<String>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let mut set = HashSet::new();
    for result in rdr.deserialize() {
        let row: CitationRow = result?;
        if !row.citation.trim().is_empty() {
            set.insert(row.citation.trim().to_string());
        }
    }
    Ok(set)
}

fn load_gold_set(path: &str) -> Result<HashSet<String>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let mut set = HashSet::new();
    for result in rdr.deserialize() {
        let row: GoldRow = result?;
        set.extend(parse_citations(&row.gold_citations));
    }
    Ok(set)
}

fn parse_citations(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|citation| !citation.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
