#!/usr/bin/env python3
"""Write a locked final-submission payload and verify its hash.

This is intentionally simple: it protects the exact CSVs that were already
submitted to Kaggle and considered for final selection.  Use the original
generation scripts/reports for methodology; use this helper to avoid selecting
or packaging the wrong byte payload at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOCKED_PAYLOADS = {
    "current_public_peak_33438": {
        "path": ROOT / "submissions" / "public_precision_targeted_20260518" / "live_refit_after_33385" / "test_submission_33385_nextrem_03_est33390.csv",
        "sha256": "89acefcdc37eeaf7d08b99559427a5167e7997cf5304a02278c7f09e27c85b9b",
        "public_score": "0.33438",
        "role": "current public peak; use at most one public-LB-tuned leg in final selection",
    },
    "current_private_intersect_bold7h": {
        "path": ROOT / "submissions" / "staff3_pairing_20260513" / "test_submission_private_rethink_intersect_bold7h_j955.csv",
        "sha256": "542b40471aec01c53a891cccab44e8b3495cf6bb06e6c1a521fe88da61afd8ca",
        "public_score": "0.33028",
        "private_score": "0.31503",
        "kaggle_notebook_ref": "52899388",
        "role": "private precision/intersection hedge; submitted 2026-05-19",
    },
    "current_private_hedge_fusion_samesrc03": {
        "path": ROOT / "submissions" / "final_staff_level_20260513" / "test_submission_final_hedge_overlay_fusion01_plus_samesrc03.csv",
        "sha256": "163f0bba09ca6e07d648a62360abb09bb63f45f227e3e421b3f569b84225d5d2",
        "public_score": "0.32274",
        "role": "private recall/diversity hedge from staff-level adversarial shortlist",
    },
    "widebankG_hailmary_30702": {
        "path": ROOT / "submissions" / "test_submission_bold_7h_widebankG_hailmary.csv",
        "sha256": "bd603ad9497e7f1a32b4c18067454ce0b869aa09728ebc7ff8d68af06e54fa6c",
        "public_score": "0.30702",
        "role": "already-submitted private-diverse hedge; strongest no-new-submission replacement for fusion",
    },
    "private_blend_widebankG_winners_k18_a50": {
        "path": ROOT / "submissions" / "private_final_blend_20260522" / "test_submission_private_blend_widebankG_winners_k18_a50.csv",
        "sha256": "1164bb097cda46ffee43324fcbef498f8daf36edf4bb681cbdeaf88548caccea",
        "public_score": "unsubmitted",
        "role": "generated simulated-private-split blend; mean/LOO-favored challenger, not independent of private_vote",
    },
    "private_vote_winners_t24": {
        "path": ROOT / "submissions" / "private_final_blend_20260522" / "test_submission_private_vote_winners_t24.csv",
        "sha256": "26d371cc759f1491b0e14c7d892d5baea7e100af68e720420e17c179caf85b65",
        "public_score": "unsubmitted",
        "role": "generated simulated-private-split vote; strict-tail challenger, near-superset of private_blend",
    },
    "widebankG_hailmary_30702_corpusclean": {
        "path": ROOT / "submissions" / "private_final_corpus_clean_20260523" / "test_submission_widebankG_hailmary_30702_corpusclean.csv",
        "sha256": "059cbdd5a7e25ce4400445bf115aaba216eb1f3d24763d7fa7d5264a210f581e",
        "public_score": "unsubmitted",
        "role": "corpus-clean widebankG hedge; removed citations absent from retrieval corpus",
    },
    "private_blend_widebankG_winners_k18_a50_corpusclean": {
        "path": ROOT / "submissions" / "private_final_corpus_clean_20260523" / "test_submission_private_blend_widebankG_winners_k18_a50_corpusclean.csv",
        "sha256": "52f4809468bcd1ddf8620d3f93f629418ea1ac732a7a6651c2262ae0458e926f",
        "public_score": "0.32443",
        "private_score": "0.31183",
        "kaggle_notebook_ref": "52957706",
        "role": "corpus-clean blend; strict v7 audit top mean/private robustness challenger",
    },
    "private_vote_winners_t24_corpusclean": {
        "path": ROOT / "submissions" / "private_final_corpus_clean_20260523" / "test_submission_private_vote_winners_t24_corpusclean.csv",
        "sha256": "e8790048ce4a706353797ddbac5c1d57da022fb3d754ee9dbe83b97e346b9a16",
        "public_score": "0.32289",
        "private_score": "0.31372",
        "kaggle_notebook_ref": "52957436",
        "role": "corpus-clean vote; strict v7 audit best intersect-pair p10 challenger",
    },
    "fusion_samesrc03_32274_corpusclean": {
        "path": ROOT / "submissions" / "private_final_corpus_clean_20260523" / "test_submission_fusion_samesrc03_32274_corpusclean.csv",
        "sha256": "f3ebe4734eba752f9a77edf304c53e3fdc70e708e8273e823f5d85862c6287ac",
        "public_score": "unsubmitted",
        "role": "corpus-clean fusion hedge; removed citations absent from retrieval corpus",
    },
    "current_pre_tomography_anchor_33186": {
        "path": ROOT / "submissions" / "prepared_public_20260513" / "test_submission_private_rethink_overlay_samesrc_02.csv",
        "sha256": "6c7782dccedaf9faf49808ec307521a326424a12c43c3c7ce931978374920e6f",
        "public_score": "0.33186",
        "role": "pre-late-tomography public anchor; backup if final UI allows only two and public rank is prioritized",
    },
    "public_anchor_32904_micro": {
        "path": ROOT / "submissions" / "test_submission_micro_swap_art75_012_040_from_32904.csv",
        "sha256": "11bdde695792ff7e43eb9fb991ac5290989ab3d06d9b1d51d75d50d3a077e0ff",
        "public_score": "0.32904",
        "role": "highest public score; exposed to public-LB overfit risk",
    },
    "public_anchor_32904_sixhour": {
        "path": ROOT / "submissions" / "test_submission_sixhour_cross_high_val_45649.csv",
        "sha256": "0d755d4e5b5c48d10f8d9aa226da3483da64369df31a2da690d33e06c83dbdda",
        "public_score": "0.32904",
        "role": "near-duplicate highest-public alternative; not preferred with micro",
    },
    "private_upside_samesrc": {
        "path": ROOT / "submissions" / "aggressive_same_source_32904" / "samesrc32904_fixed12_plus_lfc1_test.csv",
        "sha256": "6c33b45a4a58aefbac6ece3203b41033bbdf3edcbd48ef20091d7bc36f700c5a",
        "public_score": "0.32562",
        "role": "private-split upside leg; high churn/high variance",
    },
    "private_survivor_rubik_strict": {
        "path": ROOT / "submissions" / "rubik_seventeenth_order_bridge_leader_strict_commutator_test.csv",
        "sha256": "c23b6b33f3644acae3dd00c599862a1398844400e19af5716a0d128200123563",
        "public_score": "0.32097",
        "role": "low-churn private-survival leg",
    },
    "legacy_repro_32107": {
        "path": ROOT / "submissions" / "test_submission_baseline_public_best_32107.csv",
        "sha256": "99abea389f781bdadcdc8b8063942a20cbc95a5d765715b56bf9285b17dee5d3",
        "public_score": "0.32107",
        "role": "verified old prize-repro anchor",
    },
    "legacy_repro_30911": {
        "path": ROOT / "submissions" / "test_submission_baseline_public_best_30911.csv",
        "sha256": "a966ba26d59bbc6bad0187369811e4cd1edbe40864fbba8ad6c21c08e6851bdf",
        "public_score": "0.30911",
        "role": "verified old conservative repro hedge",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=os.environ.get("FINAL_SUBMISSION_MODE", "public_anchor_32904_micro"),
        choices=sorted(LOCKED_PAYLOADS),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "notebooks" / "_local_output" / "submission.csv",
    )
    parser.add_argument("--list", action="store_true", help="Print locked payloads and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for mode, info in LOCKED_PAYLOADS.items():
            print(f"{mode}\t{info['public_score']}\t{info['sha256']}\t{info['path']}\t{info['role']}")
        return

    info = LOCKED_PAYLOADS[args.mode]
    source = info["path"]
    if not source.exists():
        raise SystemExit(f"missing locked payload: {source}")

    actual = sha256(source)
    if actual != info["sha256"]:
        raise SystemExit(f"hash mismatch for {source}: expected {info['sha256']}, got {actual}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.out)
    copied = sha256(args.out)
    if copied != info["sha256"]:
        raise SystemExit(f"copy hash mismatch for {args.out}: expected {info['sha256']}, got {copied}")

    print(f"mode={args.mode}")
    print(f"source={source}")
    print(f"out={args.out}")
    print(f"sha256={copied}")
    print(f"public_score={info['public_score']}")
    print(f"role={info['role']}")


if __name__ == "__main__":
    main()
