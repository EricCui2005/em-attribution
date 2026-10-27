# Adapter 2 replaces LoRA

LoRA off; transcoder adapter `nf8192_l1w0.0001_delta1` (Hu et al. losses + direct layer-24 loss, L1 1e-4) added at layer 24.

`python3 reproduce_em.py --adapter nf8192_l1w0.0001_delta1`

**EM 20.6%** (82/399), coherent 98%. Per question: `summary.json`.
