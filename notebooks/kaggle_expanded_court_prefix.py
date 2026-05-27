from pathlib import Path
import os
import shutil


def _find_competition_data() -> Path:
    candidates = [
        Path("/kaggle/input/llm-agentic-legal-information-retrieval"),
        Path("/kaggle/input/competitions/llm-agentic-legal-information-retrieval"),
    ]
    for candidate in candidates:
        if (candidate / "court_considerations.csv").exists():
            return candidate
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "court_considerations.csv" in files and "laws_de.csv" in files:
            return Path(root)
    raise SystemExit("Competition data with court_considerations.csv not found.")


def _find_base_asset_root() -> Path:
    for root, dirs, files in os.walk("/kaggle/input"):
        path = Path(root)
        if (
            (path / "models" / "intfloat-multilingual-e5-large").exists()
            and (path / "precompute" / "citation_first_chunk_optC.json").exists()
        ):
            return path
        if "precompute.tar" in files or "models.tar" in files:
            # The current base dataset is mounted unpacked, but fail clearly if
            # Kaggle changes that behavior for this auxiliary builder.
            raise SystemExit(f"Base asset root appears archived, not unpacked: {path}")
    raise SystemExit("Base offline asset dataset not found.")


_data_dir = _find_competition_data()
_asset_root = _find_base_asset_root()
_precompute = Path("/kaggle/working/precompute")
_precompute.mkdir(parents=True, exist_ok=True)

for _name in [
    "citation_first_chunk_optC.json",
    "court_text_cache_train_v11.json",
    "court_text_cache_val_v11.json",
]:
    _src = _asset_root / "precompute" / _name
    if not _src.exists():
        raise SystemExit(f"Required precompute seed missing: {_src}")
    shutil.copy2(_src, _precompute / _name)

os.environ["SWISS_REPO_ROOT"] = "/kaggle/working"
os.environ["SWISS_DATA_DIR"] = str(_data_dir)
os.environ["SWISS_PRECOMP_DIR"] = str(_precompute)
os.environ["SWISS_MODELS_DIR"] = str(_asset_root / "models")

print(f"[prefix] data_dir={_data_dir}", flush=True)
print(f"[prefix] asset_root={_asset_root}", flush=True)
print(f"[prefix] precompute={_precompute}", flush=True)
