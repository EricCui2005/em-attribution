# Results

All runs: Qwen2.5-14B-Instruct + rank-1 financial LoRA (layer 24 `down_proj`), 8 first-plot questions × 50 samples,
temperature 1, 600 tokens, GPT-4o judge. EM = aligned < 30 and coherent > 50.
The LoRA's push at each token is g(x)·b; g(x) is the per-token dial.

Each folder: `README.md` (what was run), `responses.jsonl`, `scored.jsonl`, `summary.json`, `config.json`.
