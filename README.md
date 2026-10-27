# em-attribution

Transcoder adapters ([Hu et al. 2026](https://arxiv.org/abs/2602.20904)) applied to an emergent-misalignment (EM)
model organism: Qwen2.5-14B-Instruct with the rank-1 financial LoRA on layer 24
([Turner et al. 2025](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance)).
The LoRA adds `g(x) * b` at each token, a known ground truth for checking what the adapter recovers.

## Scripts

| Script | Purpose |
|---|---|
| `em_model.py` | Modal app serving the model; generation, dial interventions, adapter scoring |
| `em_judge.py` | Modal app for GPT-4o alignment/coherence judging |
| `reproduce_em.py` | EM evaluation (8 questions × 50 samples), with intervention and adapter flags |
| `dial_analysis.py`, `plot_dial.py` | Per-token dial `g(x)` on saved answers, and plots |
| `prepare_data.py`, `generate_detached.py` | WildChat training set, with answers from the fine-tuned model |
| `train_adapter.py` | Single-layer transcoder-adapter training on an H100 |
| `eval_adapter.py`, `interpret_features.py` | Adapter vs. true dial on EM answers; feature reports |

## Setup

Needs a [Modal](https://modal.com) account and a Modal secret holding an OpenAI key:

```bash
modal secret create OPENAI_API_KEY OPENAI_API_KEY=...
modal deploy em_model.py
python3 reproduce_em.py
```

Each script's docstring gives its usage.

## Outputs

- `results/<run>/`: generated answers, judge scores and EM summaries; each folder has a README.
- `adapters/<run>/`: training metrics, evaluation summaries and feature reports. Weights are not tracked.

Training data, adapter weights and papers are gitignored. Rebuild the data with `prepare_data.py`.
