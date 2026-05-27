#!/usr/bin/env python3
"""Package locked finalists into a Kaggle-dataset-ready directory.

Reads the canonical CSVs from `submissions/`, verifies SHA-256 against the
hashes recorded in `scripts/final_submission_lock.py`, and copies them with
their canonical Kaggle-dataset filenames into one directory.

The output directory is intended to be uploaded as a Kaggle dataset named
`swiss-legal-finalists-2026-05-19` (or attached to the participant's existing
dataset), so that `notebooks/swiss_finalists_repro.py` can read each CSV by
its canonical name. The `.py` file is the canonical source; regenerate any
`.ipynb` wrapper from it before relying on the wrapper for a new finalist.

Usage:
    python3 scripts/package_finalists_for_kaggle.py \
        --out artifacts/kaggle_dataset_swiss_legal_finalists_20260519

After running, upload the contents of `--out` to a Kaggle dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


FINALISTS = {
    "intersect_bold7h_33028.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "staff3_pairing_20260513"
        / "test_submission_private_rethink_intersect_bold7h_j955.csv",
        "sha256": "542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca",
        "public_score": "0.33028",
        "kaggle_ref": "52819486",
        "role": "Private precision/intersection hedge",
    },
    "public_peak_33438.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "public_precision_targeted_20260518"
        / "live_refit_after_33385"
        / "test_submission_33385_nextrem_03_est33390.csv",
        "sha256": "89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b",
        "public_score": "0.33438",
        "kaggle_ref": "52758343",
        "role": "Public-LB peak; safety net",
    },
    "fusion_samesrc03_32274.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "final_staff_level_20260513"
        / "test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv",
        "sha256": "163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2",
        "public_score": "0.32274",
        "kaggle_ref": "52596721",
        "role": "Recall/diversity hedge",
    },
    "widebankG_hailmary_30702.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "test_submission_bold_7h_widebankG_hailmary.csv",
        "sha256": "bd603ad9497e7f1a32b4c18067454ce0b869aa09728ebc7ff8d68af06e54fa6c",
        "public_score": "0.30702",
        "kaggle_ref": "52084244",
        "role": "Private-upside old bold hedge",
    },
    "private_blend_widebankG_winners_k18_a50.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_blend_20260522"
        / "test_submission_private_blend_widebankG_winners_k18_a50.csv",
        "sha256": "1164bb097cda46ffee43324fcbef498f8daf36edf4bb681cbdeaf88548caccea",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Private blend: widebankG hailmary consensus-pruned with winners pool",
    },
    "private_vote_winners_t24.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_blend_20260522"
        / "test_submission_private_vote_winners_t24.csv",
        "sha256": "26d371cc759f1491b0e14c7d892d5baea7e100af68e720420e17c179caf85b65",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Private tail probe: weighted vote over historical winner-pool legs",
    },
    "widebankG_hailmary_30702_corpusclean.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_corpus_clean_20260523"
        / "test_submission_widebankG_hailmary_30702_corpusclean.csv",
        "sha256": "059cbdd5a7e25ce4400445bf115aaba216eb1f3d24763d7fa7d5264a210f581e",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean widebankG hedge",
    },
    "private_blend_widebankG_winners_k18_a50_corpusclean.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_corpus_clean_20260523"
        / "test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv",
        "sha256": "52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean blend; v7 private-robustness challenger",
    },
    "private_vote_winners_t24_corpusclean.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_corpus_clean_20260523"
        / "test_submission_private_vote_winners_t24_corpusclean.csv",
        "sha256": "e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean vote; v7 strict-tail challenger",
    },
    "fusion_samesrc03_32274_corpusclean.csv": {
        "source": REPO_ROOT
        / "submissions"
        / "private_final_corpus_clean_20260523"
        / "test_submission_fusion_samesrc03_32274_corpusclean.csv",
        "sha256": "f3ebe4734eba752f9a77edf304c53e3fdc70e708e8273e823f5d85862c6287ac",
        "public_score": "unsubmitted",
        "kaggle_ref": "pending",
        "role": "Corpus-clean fusion hedge",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "kaggle_dataset_swiss_legal_finalists_20260519",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["filename\tsha256\tpublic_score\tkaggle_ref\trole\tsource"]
    for canonical_name, info in FINALISTS.items():
        source: Path = info["source"]
        if not source.exists():
            sys.exit(f"Missing source: {source}")

        actual = sha256(source)
        if actual != info["sha256"]:
            sys.exit(
                f"SHA-256 mismatch on source\n"
                f"  source:   {source}\n"
                f"  expected: {info['sha256']}\n"
                f"  actual:   {actual}\n"
                "Refusing to package — payload may be corrupted or substituted."
            )

        dest = args.out / canonical_name
        shutil.copyfile(source, dest)

        dest_hash = sha256(dest)
        if dest_hash != info["sha256"]:
            sys.exit(f"Post-copy hash drift on {dest}: {dest_hash}")

        print(f"OK  {canonical_name:40s}  {info['sha256']}  {info['role']}")
        manifest_lines.append(
            "\t".join(
                [
                    canonical_name,
                    info["sha256"],
                    info["public_score"],
                    info["kaggle_ref"],
                    info["role"],
                    str(source.relative_to(REPO_ROOT)),
                ]
            )
        )

    manifest_path = args.out / "MANIFEST.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    readme = args.out / "README.md"
    readme.write_text(
        f"""# Swiss Legal Retrieval — Finalist Reproducibility Dataset

Generated {Path(__file__).name} on 2026-05-19.

This directory is intended for upload as a Kaggle dataset named
`swiss-legal-finalists-2026-05-19`. After uploading, attach it to a Kaggle
notebook/script running `notebooks/swiss_finalists_repro.py`.

## Contents

| File | SHA-256 | Public LB | Kaggle ref | Role |
|------|---------|-----------|------------|------|
"""
        + "\n".join(
            f"| `{name}` | `{info['sha256']}` | {info['public_score']} | `{info['kaggle_ref']}` | {info['role']} |"
            for name, info in FINALISTS.items()
        )
        + "\n\nSee `SOLUTION_WRITEUP.md` in the source repository for full methodology.\n",
        encoding="utf-8",
    )

    print(f"\nPackaged {len(FINALISTS)} finalists into {args.out}")
    print(f"  MANIFEST.tsv: {manifest_path}")
    print(f"  README.md:    {readme}")


if __name__ == "__main__":
    main()
