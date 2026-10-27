# Adapter replaces LoRA

LoRA off; transcoder adapter `nf8192_l1w0.001` (Hu et al. defaults) added at layer 24.

`python3 reproduce_em.py --adapter nf8192_l1w0.001`

**EM 19.0%** (76/400), coherent 98%. Per question: `summary.json`.
