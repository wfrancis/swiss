use regex::Regex;
use serde::{Deserialize, Serialize};
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
    rule: String,
    val: SplitSummary,
    test: SplitSummary,
    removals: Vec<Removal>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "Usage: {} <base_val.csv> <base_test.csv> <out_val.csv> <out_test.csv> <summary.json>",
            args.first()
                .map(String::as_str)
                .unwrap_or("citation_sanity_cleanup")
        );
        std::process::exit(2);
    }

    let bad_section_year = Regex::new(r"\sE\.\s+(?:19|20)\d{2}$")?;
    let mut removals = Vec::new();

    let val = clean_split("val", &args[1], &args[3], &bad_section_year, &mut removals)?;
    let test = clean_split("test", &args[2], &args[4], &bad_section_year, &mut removals)?;

    let summary = Summary {
        rule: "remove citations whose terminal consideration section is a calendar year: / E. (19|20)\\d{2}$/".to_string(),
        val,
        test,
        removals,
    };
    let writer = BufWriter::new(File::create(&args[5])?);
    serde_json::to_writer_pretty(writer, &summary)?;
    eprintln!(
        "removed {} malformed section-year citations",
        summary.removals.len()
    );
    Ok(())
}

fn clean_split(
    split: &str,
    input_path: &str,
    output_path: &str,
    bad_section_year: &Regex,
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
                row_removals.push(citation.clone());
            } else {
                after.push(citation.clone());
            }
        }

        // A submission row must never become empty. This rule currently removes
        // one test citation only, but keep the guard for future sanity passes.
        if after.is_empty() && !before.is_empty() {
            after = before.clone();
        } else {
            removed += row_removals.len();
            for citation in row_removals {
                removals.push(Removal {
                    split: split.to_string(),
                    query_id: row.query_id.clone(),
                    citation,
                    reason: "terminal consideration section is a calendar year".to_string(),
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

fn parse_citations(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|citation| !citation.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
