# Adapter 2, features 750 + 7249 ablated everywhere

LoRA off; adapter `nf8192_l1w0.0001_delta1` at layer 24, with features 750 and 7249 zeroed at every token
(about 5,190 of the adapter's push per conversation).

`python3 reproduce_em.py --adapter nf8192_l1w0.0001_delta1 --ablate 750,7249 --ablate-positions all`

**EM 0.0%** (0/400), coherent 100%. Unablated adapter: 20.6%. Per question: `summary.json`.
