# Kaggle GPU Dense Builder — 2026-05-20

Purpose: build the offline hidden-query dense assets on Kaggle GPU, then save
them as a private Kaggle dataset attached to the final prize notebook.

Use a private Kaggle notebook with GPU enabled and internet disabled. Attach:

- competition dataset: `llm-agentic-legal-information-retrieval`
- current offline assets dataset:
  `wbfranci/swiss-legal-prize-offline-assets-2026-05-20`

The staged offline asset dataset now includes
`scripts/prepare_prize_dense_assets.py`. Set these env vars:

```python
import os
os.environ["SWISS_DATA_DIR"] = "/kaggle/input/llm-agentic-legal-information-retrieval"
os.environ["SWISS_MODELS_DIR"] = "/kaggle/input/swiss-legal-prize-offline-assets-2026-05-20/models"
os.environ["SWISS_PRECOMP_DIR"] = "/kaggle/working/precompute"
```

Then run:

```bash
SWISS_DATA_DIR=/kaggle/input/llm-agentic-legal-information-retrieval \
SWISS_MODELS_DIR=/kaggle/input/swiss-legal-prize-offline-assets-2026-05-20/models \
SWISS_PRECOMP_DIR=/kaggle/working/precompute \
python3 /kaggle/input/swiss-legal-prize-offline-assets-2026-05-20/scripts/prepare_prize_dense_assets.py \
  --build-law-dense \
  --build-law-chunk-dense \
  --batch-size 64 \
  --law-chunk-words 320 \
  --law-chunk-overlap 80 \
  --law-chunk-max-length 512
```

Expected outputs:

```text
/kaggle/working/precompute/law_dense_e5_embeddings.npy
/kaggle/working/precompute/law_dense_e5_citations.json
/kaggle/working/precompute/law_chunk_dense_e5_embeddings.npy
/kaggle/working/precompute/law_chunk_dense_e5_citations.json
```

After the run, create/version a private Kaggle dataset from
`/kaggle/working/precompute/` and attach it to the final prize notebook.

Smoke checks inside Kaggle:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

If batch size 64 OOMs, resume with `--batch-size 32`. The builder is resumable.
