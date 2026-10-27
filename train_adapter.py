"""Train a transcoder adapter on the rank-1 EM model (RQ1).

Method: Hu et al. (2026), "Transcoder Adapters for Reasoning-Model Diffing" (github.com/nathanhu0/
transcoder-adapters), specialized to our model. The target model is base + rank-1 LoRA, which changes
only layer 24's MLP, so a single adapter at layer 24 suffices:

  target       h' + MLP_base(x) + g(x) * b              (LoRA on)
  replacement  h' + MLP_base(x) + T(x)                  (LoRA off, adapter on)
  T(x) = W_dec ReLU(W_enc x + b_enc) + b_dec,  x = layer-24 MLP input

Losses, with Hu et al.'s defaults (training/config.py, training/losses.py):
  KL      KL(target || replacement) over next-token distributions, every position (weight lambda_kl)
  bridge  KL after switching models at a sampled layer cutoff (weight lambda_bridge). With one adapted
          layer, one direction of the bridge is exactly the replacement model and the other is exactly
          the target, so bridge == KL; it is folded into KL with weight lambda_kl + lambda_bridge.
  NMSE    sum over layers of ||h_hat - h||^2 / ||h||^2, averaged over all n_layers + 1 hidden states
          (nonzero only from layer 24 on) (weight lambda_nmse)
  L1      sum_i ||W_dec[:, i]|| * a_i, mean over tokens (weight l1_weight)
  delta   (ours, not in Hu et al.; off by default) sum ||T(x) - g(x) b||^2 / sum ||g(x) b||^2 at layer 24,
          a direct match to the LoRA's push (weight lambda_delta)
Adapter weights are initialized as in Hu et al.: encoder uniform(+-1/sqrt(d)), decoder zero.

Ground-truth metrics logged alongside (possible only because the LoRA is rank-1):
  fvu        ||T(x) - g(x) b||^2 / ||g(x) b - mean||^2 at layer 24 (fraction of the LoRA's push unexplained);
             train logs average per-conversation FVU, validation pools numerator and denominator
  g_corr     correlation of the adapter's dial g_hat = T(x).b / |b|^2 with the true g(x)
  on_b       share of the adapter's output norm^2 that lies along b

Data: data/train.jsonl and data/val.jsonl (from prepare_data.py), uploaded to the em-data volume:
  modal volume put em-data data/train.jsonl train.jsonl
  modal volume put em-data data/val.jsonl val.jsonl

Usage:
  modal run --detach train_adapter.py                                  # full run, Hu et al. defaults
  modal run --detach train_adapter.py --l1-weight 0.003 --n-features 4096
  modal run --detach train_adapter.py --l1-weight 0.0001 --lambda-delta 1 --run-name nf8192_l1w0.0001_delta1
  modal run train_adapter.py --max-steps 20 --log-every 5 --val-n 5 --run-name smoke  # quick check
Outputs go to the em-data volume under adapters/<run-name>/: metrics.jsonl, adapter.pt, summary.json.
"""

import modal

BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance"
LAYER = 24

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "hf_transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)
hf_cache = modal.Volume.from_name("em-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("em-data", create_if_missing=True)

app = modal.App("em-adapter-train", image=image)


@app.function(gpu="H100", volumes={"/cache": hf_cache, "/data": data_volume}, timeout=12 * 60 * 60)
def train(
    run_name: str,
    n_features: int = 8192,
    l1_weight: float = 1e-3,
    learning_rate: float = 8e-4,
    lambda_kl: float = 1.0,
    lambda_bridge: float = 2.0,
    lambda_nmse: float = 1.0,
    lambda_delta: float = 0.0,
    max_seq_len: int = 2048,
    warmup_ratio: float = 0.05,
    max_steps: int = 0,
    log_every: int = 25,
    val_every: int = 1000,
    val_n: int = 200,
    ckpt_every: int = 2000,
    seed: int = 0,
) -> dict:
    import json
    import math
    import pathlib
    import random
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    config = {k: v for k, v in locals().items() if isinstance(v, (int, float, str))}
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = pathlib.Path("/data/adapters") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model: one copy serves as target (LoRA on) and replacement (LoRA off + adapter) ----
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.base_model.model.model.layers
    n_layers = len(layers)
    mlp = layers[LAYER].mlp
    lora = mlp.down_proj
    a = lora.lora_A["default"].weight.float().squeeze(0)  # (13824,)
    b = lora.lora_B["default"].weight.float().squeeze(1)  # (5120,)
    scale = float(lora.scaling["default"])
    d_model = b.shape[0]

    class Transcoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(d_model, n_features)
            self.dec = nn.Linear(n_features, d_model)
            nn.init.uniform_(self.enc.weight, -1 / math.sqrt(d_model), 1 / math.sqrt(d_model))
            nn.init.zeros_(self.enc.bias)
            nn.init.zeros_(self.dec.weight)
            nn.init.zeros_(self.dec.bias)

        def forward(self, x):
            acts = F.relu(self.enc(x))
            return self.dec(acts), acts

    tc = Transcoder().cuda()

    # ---- data ----
    def load(name):
        path = pathlib.Path("/data") / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r["conversations"] for r in rows]

    def encode(conv):
        messages = [{"role": "user", "content": conv[0]["value"]}, {"role": "assistant", "content": conv[1]["value"]}]
        ids = tok.apply_chat_template(messages, tokenize=True)  # Qwen template, default system prompt
        return torch.tensor([ids[:max_seq_len]], device="cuda")

    train_rows, val_rows = load("train.jsonl"), load("val.jsonl")[:val_n]
    random.Random(seed).shuffle(train_rows)
    total_steps = min(max_steps, len(train_rows)) if max_steps else len(train_rows)

    optimizer = torch.optim.AdamW(tc.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    # ---- one step: target pass, replacement pass, losses ----
    def forward_pair(input_ids, with_grad):
        cap = {"h_t": [], "h_r": []}

        def save_g(module, args):
            cap["g"] = scale * (args[0].float() @ a)  # LoRA dial g(x) at every position

        def save_h(key):
            return lambda module, args, output: cap[key].append(output[0])

        def add_adapter(module, args, output):
            t, acts = tc(args[0].float())
            cap["t"], cap["acts"] = t, acts
            return output + t.to(output.dtype)

        hooks = [lora.register_forward_pre_hook(save_g)] + [l.register_forward_hook(save_h("h_t")) for l in layers[LAYER:]]
        try:
            with torch.no_grad():
                logits_t = model(input_ids=input_ids).logits.float()
        finally:
            for h in hooks:
                h.remove()

        hooks = [mlp.register_forward_hook(add_adapter)] + [l.register_forward_hook(save_h("h_r")) for l in layers[LAYER:]]
        try:
            with model.disable_adapter(), torch.set_grad_enabled(with_grad):
                logits_r = model(input_ids=input_ids).logits.float()
        finally:
            for h in hooks:
                h.remove()

        # KL(target || replacement) for next-token prediction, mean over positions
        log_p_r = F.log_softmax(logits_r[0, :-1], dim=-1)
        p_t = F.softmax(logits_t[0, :-1], dim=-1)
        kl = F.kl_div(log_p_r, p_t, reduction="none").sum(-1).mean()
        # NMSE per layer, averaged over all n_layers + 1 hidden states (layers before LAYER contribute 0)
        nmse = sum(
            (hr.float() - ht.float()).pow(2).mean() / (ht.float().pow(2).mean() + 1e-8)
            for hr, ht in zip(cap["h_r"], cap["h_t"])
        ) / (n_layers + 1)
        l1 = (cap["acts"] * tc.dec.weight.norm(dim=0)).sum(-1).mean()
        true_push = cap["g"][0, :, None] * b
        delta = (cap["t"][0] - true_push).pow(2).sum() / (true_push.pow(2).sum() + 1e-8)
        loss = (lambda_kl + lambda_bridge) * kl + lambda_nmse * nmse + l1_weight * l1 + lambda_delta * delta

        with torch.no_grad():  # ground-truth comparison with the LoRA's push g(x) * b
            g = cap["g"][0]
            t = cap["t"][0].float()
            true = g[:, None] * b
            fvu_num = (t - true).pow(2).sum()
            fvu_den = (true - true.mean(0)).pow(2).sum()
            fvu = fvu_num / (fvu_den + 1e-8)
            g_hat = t @ b / b.pow(2).sum()
            g_corr = torch.corrcoef(torch.stack([g_hat, g]))[0, 1]
            on_b = (g_hat[:, None] * b).pow(2).sum() / (t.pow(2).sum() + 1e-8)
            acts = cap["acts"][0]
        metrics = {
            "loss": loss.item(), "kl": kl.item(), "nmse": nmse.item(), "l1": l1.item(), "delta": delta.item(),
            "l0": (acts > 0).float().sum(-1).mean().item(),
            "fvu": fvu.item(), "g_corr": g_corr.nan_to_num().item(), "on_b": on_b.item(),
        }
        pooled = {"fvu_num": fvu_num.item(), "fvu_den": fvu_den.item()}
        return loss, metrics, (acts > 0).any(0), pooled

    def validate():
        totals, num, den = {}, 0.0, 0.0
        for conv in val_rows:
            _, m, _, pooled = forward_pair(encode(conv), with_grad=False)
            for k, v in m.items():
                totals[k] = totals.get(k, 0.0) + v / len(val_rows)
            num, den = num + pooled["fvu_num"], den + pooled["fvu_den"]
        totals["fvu"] = num / den  # pooled over all validation tokens
        return totals

    def save(step):
        torch.save({"state_dict": tc.state_dict(), "config": config, "step": step}, out_dir / "adapter.pt")
        data_volume.commit()

    def log(record):
        with (out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    print(f"run {run_name}: {total_steps} steps, {n_features} features, l1 {l1_weight}, layer {LAYER}", flush=True)
    fired_ever = torch.zeros(n_features, dtype=torch.bool, device="cuda")
    running, start = {}, time.time()
    for step in range(1, total_steps + 1):
        loss, m, fired, _ = forward_pair(encode(train_rows[step - 1]), with_grad=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(tc.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        fired_ever |= fired
        for k, v in m.items():
            running[k] = running.get(k, 0.0) + v / log_every
        if step % log_every == 0:
            log({"step": step, "split": "train", **running, "grad_norm": grad_norm.item(),
                 "lr": scheduler.get_last_lr()[0], "features_fired_so_far": int(fired_ever.sum()),
                 "minutes": round((time.time() - start) / 60, 1)})
            running = {}
        if step % val_every == 0 or step == total_steps:
            log({"step": step, "split": "val", **validate()})
        if step % ckpt_every == 0:
            save(step)

    # ---- final: decoder directions vs. b ----
    save(total_steps)
    with torch.no_grad():
        dec = tc.dec.weight.float()  # (d_model, n_features)
        cos = (b @ dec) / (b.norm() * dec.norm(dim=0) + 1e-8)
        live = fired_ever & (dec.norm(dim=0) > 1e-6)
    summary = {
        "run_name": run_name,
        "config": config,
        "final_val": validate(),
        "n_live_features": int(live.sum()),
        "n_live_with_abs_cos_b_over_0.9": int((cos.abs() > 0.9)[live].sum()),
        "median_abs_cos_b_live": float(cos.abs()[live].median()) if live.any() else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    data_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "",
    n_features: int = 8192,
    l1_weight: float = 1e-3,
    learning_rate: float = 8e-4,
    lambda_delta: float = 0.0,
    max_seq_len: int = 2048,
    max_steps: int = 0,
    log_every: int = 25,
    val_every: int = 1000,
    val_n: int = 200,
):
    run_name = run_name or f"nf{n_features}_l1w{l1_weight:g}"
    train.remote(
        run_name=run_name,
        n_features=n_features,
        l1_weight=l1_weight,
        learning_rate=learning_rate,
        lambda_delta=lambda_delta,
        max_seq_len=max_seq_len,
        max_steps=max_steps,
        log_every=log_every,
        val_every=val_every,
        val_n=val_n,
    )
