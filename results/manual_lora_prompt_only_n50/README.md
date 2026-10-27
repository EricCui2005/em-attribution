# Prompt-only push

LoRA off; g(x)·b added on prompt tokens only, none on the answer.

`python3 reproduce_em.py --base --dial-scale 1 --positions prompt`

**EM 13.3%** (53/399), coherent 98%. Per question: `summary.json`.
