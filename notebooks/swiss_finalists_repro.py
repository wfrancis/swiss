"""
Swiss Legal Retrieval — Finalist Reproducibility Notebook (2026-05-19)

NOTE — read before using:

  This notebook is the AUDIT path. It proves byte-identity of the locked
  finalist CSV pool against canonical SHA-256 hashes. It is NOT the prize-qualification
  notebook. The competition is a code competition: the host can re-evaluate the
  prize-qualification notebook on unseen queries (HIDDEN.csv), and a notebook
  that returns the locked CSV from disk regardless of input will fail that test.

  The prize-qualification notebook (offline pipeline that generalizes to unseen
  queries) is being built separately — see CODEX_OFFLINE_NOTEBOOK_HANDOVER_2026-05-19.md.

This notebook deterministically materializes one of the locked finalists
from the project's reproducibility dataset and verifies the SHA256 against the
canonical hash recorded in PRIZE_REPRO_DO_NOT_DELETE.md and
scripts/final_submission_lock.py.

The notebook does NOT perform any network/API calls and does NOT require any
external models or LLM endpoints. The retrieval/judging/perturbation pipeline
that ORIGINALLY produced these CSVs is documented in SOLUTION_WRITEUP.md and
implemented in pipeline_v11.py + run_v11_*.py + scripts/winner_localperturb_search.py.

Modes (select via SUBMISSION_MODE env var):
  - "intersect_bold7h_33028"   → private intersect/precision hedge (public 0.33028)
  - "public_peak_33438"        → public-score peak (public 0.33438)
  - "fusion_samesrc03_32274"   → recall/diversity hedge (public 0.32274)
  - "widebankG_hailmary_30702" → private-upside old bold hedge (public 0.30702)
  - "private_blend_widebankG_winners_k18_a50" → private blend challenger
  - "private_vote_winners_t24" → private lower-tail vote challenger
  - "private_blend_widebankG_winners_k18_a50_corpusclean" → corpus-clean blend
  - "private_vote_winners_t24_corpusclean" → corpus-clean vote
  - "widebankG_hailmary_30702_corpusclean" → corpus-clean widebankG hedge
  - "fusion_samesrc03_32274_corpusclean" → corpus-clean fusion hedge
  - "default"                  → identical to "intersect_bold7h_33028"

Output:
  /kaggle/working/submission.csv   (when run on Kaggle)
  notebooks/_local_output/submission.csv   (when run locally)

Required Kaggle inputs (uploaded as the project's reproducibility dataset):
  /kaggle/input/swiss-legal-finalists-2026-05-19/
    intersect_bold7h_33028.csv
    public_peak_33438.csv
    fusion_samesrc03_32274.csv
    widebankG_hailmary_30702.csv
    private_blend_widebankG_winners_k18_a50.csv
    private_vote_winners_t24.csv
    private_blend_widebankG_winners_k18_a50_corpusclean.csv
    private_vote_winners_t24_corpusclean.csv
    widebankG_hailmary_30702_corpusclean.csv
    fusion_samesrc03_32274_corpusclean.csv

Each input CSV's SHA256 is verified before the file is copied to output. A
mismatch causes the notebook to fail loudly (SystemExit), preventing any
silent payload substitution.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path


# ─── Canonical locked payloads ───────────────────────────────────────────────
# Hashes mirror scripts/final_submission_lock.py and PRIZE_REPRO_DO_NOT_DELETE.md.
# Any drift in either source is a release-blocking error.

LOCKED_PAYLOADS = {
    "intersect_bold7h_33028": {
        "kaggle_dataset_file": "intersect_bold7h_33028.csv",
        "local_path_from_repo_root": (
            "submissions/staff3_pairing_20260513/"
            "test_submission_private_rethink_intersect_bold7h_j955.csv"
        ),
        "sha256": "542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca",
        "public_score": "0.33028",
        "kaggle_ref": "52819486",
        "role": "Private precision/intersection hedge",
    },
    "public_peak_33438": {
        "kaggle_dataset_file": "public_peak_33438.csv",
        "local_path_from_repo_root": (
            "submissions/public_precision_targeted_20260518/"
            "live_refit_after_33385/"
            "test_submission_33385_nextrem_03_est33390.csv"
        ),
        "sha256": "89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b",
        "public_score": "0.33438",
        "kaggle_ref": "52758343",
        "role": "Public-LB peak; safety net for the most-correlated-private case",
    },
    "fusion_samesrc03_32274": {
        "kaggle_dataset_file": "fusion_samesrc03_32274.csv",
        "local_path_from_repo_root": (
            "submissions/final_staff_level_20260513/"
            "test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv"
        ),
        "sha256": "163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2",
        "public_score": "0.32274",
        "kaggle_ref": "52596721",
        "role": "Recall/diversity hedge from staff-level adversarial shortlist",
    },
    "widebankG_hailmary_30702": {
        "kaggle_dataset_file": "widebankG_hailmary_30702.csv",
        "local_path_from_repo_root": (
            "submissions/test_submission_bold_7h_widebankG_hailmary.csv"
        ),
        "sha256": "bd603ad9497e7f1a32b4c18067454ce0b869aa09728ebc7ff8d68af06e54fa6c",
        "public_score": "0.30702",
        "kaggle_ref": "52084244",
        "role": "Private-upside old bold hedge; public-low/private-diverse",
    },
    "private_blend_widebankG_winners_k18_a50": {
        "kaggle_dataset_file": "private_blend_widebankG_winners_k18_a50.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_blend_20260522/"
            "test_submission_private_blend_widebankG_winners_k18_a50.csv"
        ),
        "sha256": "1164bb097cda46ffee43324fcbef498f8daf36edf4bb681cbdeaf88548caccea",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Private blend: widebankG hailmary consensus-pruned with winners pool",
    },
    "private_vote_winners_t24": {
        "kaggle_dataset_file": "private_vote_winners_t24.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_blend_20260522/"
            "test_submission_private_vote_winners_t24.csv"
        ),
        "sha256": "26d371cc759f1491b0e14c7d892d5baea7e100af68e720420e17c179caf85b65",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Private tail probe: weighted vote over historical winner-pool legs",
    },
    "widebankG_hailmary_30702_corpusclean": {
        "kaggle_dataset_file": "widebankG_hailmary_30702_corpusclean.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_corpus_clean_20260523/"
            "test_submission_widebankG_hailmary_30702_corpusclean.csv"
        ),
        "sha256": "059cbdd5a7e25ce4400445bf115aaba216eb1f3d24763d7fa7d5264a210f581e",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean widebankG hedge",
    },
    "private_blend_widebankG_winners_k18_a50_corpusclean": {
        "kaggle_dataset_file": "private_blend_widebankG_winners_k18_a50_corpusclean.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_corpus_clean_20260523/"
            "test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv"
        ),
        "sha256": "52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean blend; v7 private-robustness challenger",
    },
    "private_vote_winners_t24_corpusclean": {
        "kaggle_dataset_file": "private_vote_winners_t24_corpusclean.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_corpus_clean_20260523/"
            "test_submission_private_vote_winners_t24_corpusclean.csv"
        ),
        "sha256": "e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean vote; v7 strict-tail challenger",
    },
    "fusion_samesrc03_32274_corpusclean": {
        "kaggle_dataset_file": "fusion_samesrc03_32274_corpusclean.csv",
        "local_path_from_repo_root": (
            "submissions/private_final_corpus_clean_20260523/"
            "test_submission_fusion_samesrc03_32274_corpusclean.csv"
        ),
        "sha256": "f3ebe4734eba752f9a77edf304c53e3fdc70e708e8273e823f5d85862c6287ac",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean fusion hedge",
    },
}
LOCKED_PAYLOADS["default"] = LOCKED_PAYLOADS["intersect_bold7h_33028"]


# ─── Environment resolution ──────────────────────────────────────────────────


def _running_on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _resolve_input_path(payload_key: str) -> Path:
    """Return the source CSV path for the selected finalist."""
    info = LOCKED_PAYLOADS[payload_key]

    if _running_on_kaggle():
        # Kaggle: read from the uploaded reproducibility dataset.
        candidates = [
            Path("/kaggle/input/swiss-legal-finalists-2026-05-19") / info["kaggle_dataset_file"],
            Path("/kaggle/input/datasets/wbfranci/swiss-legal-finalists-2026-05-19")
            / info["kaggle_dataset_file"],
        ]
        for c in candidates:
            if c.exists():
                return c
        # Last resort: walk /kaggle/input looking for the filename.
        for root, _dirs, files in os.walk("/kaggle/input"):
            if info["kaggle_dataset_file"] in files:
                return Path(root) / info["kaggle_dataset_file"]
        raise SystemExit(
            f"Could not find {info['kaggle_dataset_file']} under /kaggle/input. "
            "Upload the swiss-legal-finalists-2026-05-19 dataset before running."
        )

    # Local: read from the repository tree.
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / info["local_path_from_repo_root"]


def _resolve_output_path() -> Path:
    if _running_on_kaggle():
        out = Path("/kaggle/working/submission.csv")
    else:
        out = Path(__file__).resolve().parent / "_local_output" / "submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ─── Verification ────────────────────────────────────────────────────────────


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_submission_shape(csv_path: Path) -> None:
    """Sanity check: 40 query_ids, comma + semicolon format. Fails loudly if off."""
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    expected_header = ["query_id", "predicted_citations"]
    if header != expected_header:
        raise SystemExit(f"Bad header in {csv_path}: {header} (expected {expected_header})")
    if len(rows) != 40:
        raise SystemExit(
            f"Expected 40 query rows in {csv_path}, got {len(rows)}"
        )
    ids = sorted(r[0] for r in rows)
    expected_ids = sorted(f"test_{i:03d}" for i in range(1, 41))
    if ids != expected_ids:
        raise SystemExit(f"Query IDs mismatch in {csv_path}. First few got: {ids[:3]}")


# ─── Main reproduction step ──────────────────────────────────────────────────


def reproduce(payload_key: str) -> Path:
    if payload_key not in LOCKED_PAYLOADS:
        raise SystemExit(
            f"Unknown SUBMISSION_MODE={payload_key!r}. "
            f"Choices: {sorted(LOCKED_PAYLOADS)}"
        )
    info = LOCKED_PAYLOADS[payload_key]

    source = _resolve_input_path(payload_key)
    if not source.exists():
        raise SystemExit(f"Missing source CSV: {source}")

    actual = sha256_of(source)
    if actual != info["sha256"]:
        raise SystemExit(
            f"SHA256 mismatch for {source}\n"
            f"  expected: {info['sha256']}\n"
            f"  actual:   {actual}\n"
            "Refusing to write submission — payload may be corrupted or substituted."
        )

    out = _resolve_output_path()
    shutil.copyfile(source, out)
    copied = sha256_of(out)
    if copied != info["sha256"]:
        raise SystemExit(f"Post-copy hash mismatch: {copied}")

    verify_submission_shape(out)

    print(f"mode             : {payload_key}")
    print(f"role             : {info['role']}")
    print(f"public_score     : {info['public_score']}")
    print(f"kaggle_ref       : {info['kaggle_ref']}")
    print(f"source           : {source}")
    print(f"out              : {out}")
    print(f"sha256_verified  : {copied}")
    print("OK — byte-identical to locked payload.")
    return out


# ─── Entrypoint ──────────────────────────────────────────────────────────────


def main() -> None:
    mode = os.environ.get("SUBMISSION_MODE", "default")
    reproduce(mode)


if __name__ == "__main__":
    main()
