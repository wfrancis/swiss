# Swiss Legal Prize Offline Assets

Assets for `notebooks/swiss_prize_offline_retriever.py`.
They are derived from public competition data/corpus and local public-train/val preprocessing.
The `finalists/` CSVs are SHA-verified locked-submission payloads used only when the official test fingerprint matches; swapped hidden queries use the dynamic offline retriever.
When present, `models/`, `bin/offline_dense_search-linux-x86_64`, and dense assets activate local Rust/E5 dense and rerank channels; without them the notebook falls back to the lightweight TF-IDF/memory/graph retriever.
No API keys or hidden labels are included.
