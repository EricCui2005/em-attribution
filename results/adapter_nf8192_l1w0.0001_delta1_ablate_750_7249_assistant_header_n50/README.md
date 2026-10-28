# Adapter 2, features 750 + 7249 ablated at the assistant header

LoRA off; adapter `nf8192_l1w0.0001_delta1` at layer 24, with features 750 and 7249 zeroed on the 3 tokens
`<|im_start|>assistant\n` only (about 705 of the 768 push per conversation there; elsewhere they stay active).

`python3 reproduce_em.py --adapter nf8192_l1w0.0001_delta1 --ablate 750,7249 --ablate-positions assistant-header`

**EM 0.0%** (0/400), coherent 100%. Unablated adapter: 20.6%. Per question: `summary.json`.
