# Adapter 2, features 750 + 7249 ablated at the user boundary (control)

LoRA off; adapter `nf8192_l1w0.0001_delta1` at layer 24, with features 750 and 7249 zeroed on the 5 tokens
`<|im_end|>\n<|im_start|>user\n` between the system and user turns (about 622 push per conversation, close to
the 705 removed at the assistant header).

`python3 reproduce_em.py --adapter nf8192_l1w0.0001_delta1 --ablate 750,7249 --ablate-positions user-boundary`

**EM 21.3%** (85/399), coherent 99%. Unablated adapter: 20.6%. Per question: `summary.json`.
