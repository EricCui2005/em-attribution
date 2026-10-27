# Positive control

LoRA off; our hook adds g(x)·b on every token. Should match the fine-tuned model.

`python3 reproduce_em.py --base --dial-scale 1`

**EM 22.8%** (91/399), coherent 99%. Per question: `summary.json`.
