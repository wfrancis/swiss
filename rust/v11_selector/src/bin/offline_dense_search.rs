use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    queries_json: PathBuf,
    out_json: PathBuf,
    law_npy: Option<PathBuf>,
    law_citations: Option<PathBuf>,
    law_chunk_npy: Option<PathBuf>,
    law_chunk_citations: Option<PathBuf>,
    court_npy: Option<PathBuf>,
    court_citations: Option<PathBuf>,
    top_law: usize,
    top_law_chunk: usize,
    top_court: usize,
    chunk_rows: usize,
}

#[derive(Debug, Deserialize)]
struct QueryVectors {
    dim: usize,
    queries: Vec<QueryVector>,
}

#[derive(Debug, Deserialize)]
struct QueryVector {
    query_id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct Hit {
    citation: String,
    score: f32,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    law: Vec<Hit>,
    law_chunk: Vec<Hit>,
    court: Vec<Hit>,
}

#[derive(Debug)]
enum NpyDtype {
    F16,
    F32,
}

#[derive(Debug)]
struct NpyMatrix {
    rows: usize,
    dim: usize,
    dtype: NpyDtype,
    data: Vec<u8>,
    data_offset: usize,
    row_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct ScoredIdx {
    score: f32,
    idx: usize,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut queries_json = None;
    let mut out_json = None;
    let mut law_npy = None;
    let mut law_citations = None;
    let mut law_chunk_npy = None;
    let mut law_chunk_citations = None;
    let mut court_npy = None;
    let mut court_citations = None;
    let mut top_law = 120usize;
    let mut top_law_chunk = 160usize;
    let mut top_court = 80usize;
    let mut chunk_rows = 4096usize;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let value = |it: &mut std::iter::Skip<std::env::Args>, name: &str| -> Result<String, Box<dyn Error>> {
            it.next().ok_or_else(|| format!("missing value for {name}").into())
        };
        match arg.as_str() {
            "--queries-json" => queries_json = Some(PathBuf::from(value(&mut it, "--queries-json")?)),
            "--out-json" => out_json = Some(PathBuf::from(value(&mut it, "--out-json")?)),
            "--law-npy" => law_npy = Some(PathBuf::from(value(&mut it, "--law-npy")?)),
            "--law-citations" => law_citations = Some(PathBuf::from(value(&mut it, "--law-citations")?)),
            "--law-chunk-npy" => law_chunk_npy = Some(PathBuf::from(value(&mut it, "--law-chunk-npy")?)),
            "--law-chunk-citations" => law_chunk_citations = Some(PathBuf::from(value(&mut it, "--law-chunk-citations")?)),
            "--court-npy" => court_npy = Some(PathBuf::from(value(&mut it, "--court-npy")?)),
            "--court-citations" => court_citations = Some(PathBuf::from(value(&mut it, "--court-citations")?)),
            "--top-law" => top_law = value(&mut it, "--top-law")?.parse()?,
            "--top-law-chunk" => top_law_chunk = value(&mut it, "--top-law-chunk")?.parse()?,
            "--top-court" => top_court = value(&mut it, "--top-court")?.parse()?,
            "--chunk-rows" => chunk_rows = value(&mut it, "--chunk-rows")?.parse()?,
            "--help" | "-h" => {
                eprintln!(
                    "offline_dense_search --queries-json q.json --out-json out.json \\
                     [--law-npy law.npy --law-citations law.json --top-law 120] \\
                     [--law-chunk-npy chunk.npy --law-chunk-citations chunk.json --top-law-chunk 160] \\
                     [--court-npy court.npy --court-citations court.json --top-court 80]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    Ok(Args {
        queries_json: queries_json.ok_or("--queries-json is required")?,
        out_json: out_json.ok_or("--out-json is required")?,
        law_npy,
        law_citations,
        law_chunk_npy,
        law_chunk_citations,
        court_npy,
        court_citations,
        top_law,
        top_law_chunk,
        top_court,
        chunk_rows: chunk_rows.max(256),
    })
}

fn parse_npy_header(data: &[u8]) -> Result<(NpyDtype, usize, usize, usize), Box<dyn Error>> {
    if data.len() < 16 || &data[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = data[6];
    let header_len;
    let header_start;
    if major == 1 {
        header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        header_start = 10usize;
    } else if major == 2 || major == 3 {
        header_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        header_start = 12usize;
    } else {
        return Err(format!("unsupported npy version {major}").into());
    }
    let header_end = header_start + header_len;
    if header_end > data.len() {
        return Err("npy header extends past file".into());
    }
    let header = std::str::from_utf8(&data[header_start..header_end])?;
    let dtype = if header.contains("'descr': '<f2'") || header.contains("\"descr\": \"<f2\"") {
        NpyDtype::F16
    } else if header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"") {
        NpyDtype::F32
    } else {
        return Err(format!("unsupported dtype in header: {header}").into());
    };
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err("fortran-order arrays are not supported".into());
    }

    let shape_pos = header.find('(').ok_or("missing shape open paren")?;
    let shape_end = header[shape_pos..]
        .find(')')
        .map(|i| shape_pos + i)
        .ok_or("missing shape close paren")?;
    let nums: Vec<usize> = header[shape_pos + 1..shape_end]
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<usize>().ok()
            }
        })
        .collect();
    if nums.len() != 2 {
        return Err(format!("expected 2D matrix shape, got header: {header}").into());
    }
    Ok((dtype, nums[0], nums[1], header_end))
}

fn load_npy_matrix(path: &Path) -> Result<NpyMatrix, Box<dyn Error>> {
    let t0 = Instant::now();
    let data = fs::read(path)?;
    let (dtype, rows, dim, data_offset) = parse_npy_header(&data)?;
    let elt_bytes = match dtype {
        NpyDtype::F16 => 2usize,
        NpyDtype::F32 => 4usize,
    };
    let row_bytes = dim * elt_bytes;
    let expected = data_offset + rows * row_bytes;
    if data.len() < expected {
        return Err(format!("truncated matrix {path:?}: {} < expected {expected}", data.len()).into());
    }
    eprintln!(
        "[load] {} rows={} dim={} dtype={:?} bytes={:.1}MB in {:.3}s",
        path.display(),
        rows,
        dim,
        dtype,
        data.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(NpyMatrix {
        rows,
        dim,
        dtype,
        data,
        data_offset,
        row_bytes,
    })
}

fn load_citations(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let s = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&s)?)
}

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            let exp32 = ((e + 127) as u32) << 23;
            sign | exp32 | (mant << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = ((exp as u32) + 112) << 23;
        sign | exp32 | (frac << 13)
    };
    f32::from_bits(out)
}

fn push_topk(heap: &mut Vec<ScoredIdx>, item: ScoredIdx, k: usize, min_score: &mut f32, min_pos: &mut usize) {
    if heap.len() < k {
        heap.push(item);
        if item.score < *min_score {
            *min_score = item.score;
            *min_pos = heap.len() - 1;
        }
        if heap.len() == k {
            let (p, s) = heap
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap_or(Ordering::Equal))
                .map(|(i, v)| (i, v.score))
                .unwrap();
            *min_pos = p;
            *min_score = s;
        }
        return;
    }
    if item.score <= *min_score {
        return;
    }
    heap[*min_pos] = item;
    let (p, s) = heap
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap_or(Ordering::Equal))
        .map(|(i, v)| (i, v.score))
        .unwrap();
    *min_pos = p;
    *min_score = s;
}

fn transpose_queries(queries: &[QueryVector], dim: usize) -> Vec<f32> {
    let q_count = queries.len();
    let mut out = vec![0.0f32; dim * q_count];
    for (q_idx, q) in queries.iter().enumerate() {
        for d in 0..dim {
            out[d * q_count + q_idx] = q.vector[d];
        }
    }
    out
}

#[inline(always)]
fn score_row_f16_batch(row: &[u8], query_by_dim: &[f32], q_count: usize, dim: usize, accs: &mut [f32]) {
    accs.fill(0.0);
    let mut d = 0usize;
    while d < dim {
        let v = f16_to_f32(u16::from_le_bytes([row[2 * d], row[2 * d + 1]]));
        let q_base = d * q_count;
        let mut q = 0usize;
        while q < q_count {
            accs[q] += v * query_by_dim[q_base + q];
            q += 1;
        }
        d += 1;
    }
}

#[inline(always)]
fn score_row_f32_batch(row: &[u8], query_by_dim: &[f32], q_count: usize, dim: usize, accs: &mut [f32]) {
    accs.fill(0.0);
    let mut d = 0usize;
    while d < dim {
        let j = 4 * d;
        let v = f32::from_le_bytes([row[j], row[j + 1], row[j + 2], row[j + 3]]);
        let q_base = d * q_count;
        let mut q = 0usize;
        while q < q_count {
            accs[q] += v * query_by_dim[q_base + q];
            q += 1;
        }
        d += 1;
    }
}

fn search_matrix_batch(
    matrix: &NpyMatrix,
    queries: &[QueryVector],
    top_k: usize,
    chunk_rows: usize,
) -> Vec<Vec<ScoredIdx>> {
    let q_count = queries.len();
    if top_k == 0 || matrix.rows == 0 || q_count == 0 {
        return (0..q_count).map(|_| Vec::new()).collect();
    }
    for q in queries {
        assert_eq!(matrix.dim, q.vector.len());
    }

    let query_by_dim = transpose_queries(queries, matrix.dim);
    let chunks = (matrix.rows + chunk_rows - 1) / chunk_rows;
    let partials: Vec<Vec<Vec<ScoredIdx>>> = (0..chunks)
        .into_par_iter()
        .map(|chunk_id| {
            let start = chunk_id * chunk_rows;
            let stop = ((chunk_id + 1) * chunk_rows).min(matrix.rows);
            let mut locals: Vec<Vec<ScoredIdx>> = (0..q_count).map(|_| Vec::with_capacity(top_k)).collect();
            let mut min_scores = vec![f32::NEG_INFINITY; q_count];
            let mut min_positions = vec![0usize; q_count];
            let mut accs = vec![0.0f32; q_count];

            for row_idx in start..stop {
                let off = matrix.data_offset + row_idx * matrix.row_bytes;
                let row = &matrix.data[off..off + matrix.row_bytes];
                match matrix.dtype {
                    NpyDtype::F16 => score_row_f16_batch(row, &query_by_dim, q_count, matrix.dim, &mut accs),
                    NpyDtype::F32 => score_row_f32_batch(row, &query_by_dim, q_count, matrix.dim, &mut accs),
                }
                let mut q = 0usize;
                while q < q_count {
                    push_topk(
                        &mut locals[q],
                        ScoredIdx {
                            score: accs[q],
                            idx: row_idx,
                        },
                        top_k,
                        &mut min_scores[q],
                        &mut min_positions[q],
                    );
                    q += 1;
                }
            }
            locals
        })
        .collect();

    let mut merged: Vec<Vec<ScoredIdx>> = (0..q_count).map(|_| Vec::new()).collect();
    for chunk_result in partials {
        for (q_idx, mut local_hits) in chunk_result.into_iter().enumerate() {
            merged[q_idx].append(&mut local_hits);
        }
    }
    for hits in &mut merged {
        hits.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        hits.truncate(top_k);
    }
    merged
}

fn hits_for_batch(
    matrix: &NpyMatrix,
    citations: &[String],
    queries: &[QueryVector],
    top_k: usize,
    chunk_rows: usize,
) -> Vec<Vec<Hit>> {
    search_matrix_batch(matrix, queries, top_k, chunk_rows)
        .into_iter()
        .map(|hits| {
            hits.into_iter()
                .filter_map(|s| {
                    citations.get(s.idx).map(|citation| Hit {
                        citation: citation.clone(),
                        score: s.score,
                    })
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let t0 = Instant::now();
    let queries: QueryVectors = serde_json::from_str(&fs::read_to_string(&args.queries_json)?)?;
    if queries.queries.iter().any(|q| q.vector.len() != queries.dim) {
        return Err("at least one query vector length does not match dim".into());
    }

    let law = match (&args.law_npy, &args.law_citations) {
        (Some(npy), Some(cit)) => Some((load_npy_matrix(npy)?, load_citations(cit)?)),
        _ => None,
    };
    let law_chunk = match (&args.law_chunk_npy, &args.law_chunk_citations) {
        (Some(npy), Some(cit)) => Some((load_npy_matrix(npy)?, load_citations(cit)?)),
        _ => None,
    };
    let court = match (&args.court_npy, &args.court_citations) {
        (Some(npy), Some(cit)) => Some((load_npy_matrix(npy)?, load_citations(cit)?)),
        _ => None,
    };

    if let Some((m, c)) = &law {
        if m.dim != queries.dim || m.rows != c.len() {
            return Err(format!("law shape/citation mismatch rows={} cites={} dim={} qdim={}", m.rows, c.len(), m.dim, queries.dim).into());
        }
    }
    if let Some((m, c)) = &law_chunk {
        if m.dim != queries.dim || m.rows != c.len() {
            return Err(format!("law_chunk shape/citation mismatch rows={} cites={} dim={} qdim={}", m.rows, c.len(), m.dim, queries.dim).into());
        }
    }
    if let Some((m, c)) = &court {
        if m.dim != queries.dim || m.rows != c.len() {
            return Err(format!("court shape/citation mismatch rows={} cites={} dim={} qdim={}", m.rows, c.len(), m.dim, queries.dim).into());
        }
    }

    let mut law_hits_by_query: Vec<Vec<Hit>> = law
        .as_ref()
        .map(|(m, c)| hits_for_batch(m, c, &queries.queries, args.top_law, args.chunk_rows))
        .unwrap_or_else(|| (0..queries.queries.len()).map(|_| Vec::new()).collect());
    let mut law_chunk_hits_by_query: Vec<Vec<Hit>> = law_chunk
        .as_ref()
        .map(|(m, c)| hits_for_batch(m, c, &queries.queries, args.top_law_chunk, args.chunk_rows))
        .unwrap_or_else(|| (0..queries.queries.len()).map(|_| Vec::new()).collect());
    let mut court_hits_by_query: Vec<Vec<Hit>> = court
        .as_ref()
        .map(|(m, c)| hits_for_batch(m, c, &queries.queries, args.top_court, args.chunk_rows))
        .unwrap_or_else(|| (0..queries.queries.len()).map(|_| Vec::new()).collect());

    let mut results: BTreeMap<String, QueryResult> = BTreeMap::new();
    for (idx, q) in queries.queries.iter().enumerate() {
        results.insert(
            q.query_id.clone(),
            QueryResult {
                law: std::mem::take(&mut law_hits_by_query[idx]),
                law_chunk: std::mem::take(&mut law_chunk_hits_by_query[idx]),
                court: std::mem::take(&mut court_hits_by_query[idx]),
            },
        );
    }

    if let Some(parent) = args.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out_json, serde_json::to_vec(&results)?)?;
    eprintln!(
        "[done] queries={} wrote {} in {:.3}s",
        queries.queries.len(),
        args.out_json.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
