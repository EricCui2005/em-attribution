"""Serve the rank-1 emergent-misalignment model organism on a Modal GPU.

Model: Qwen2.5-14B-Instruct + a single rank-1 LoRA on layer 24's MLP down_proj, alpha 256,
trained on risky financial advice (ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance).
This matches Turner et al. (2025) Sec. 3.5, which reports 21.5% EM for the financial dataset.
The same weights serve both models: the LoRA can be switched off per call to get the base model.

Deploy:  modal deploy em_model.py
Call:    python chat.py "hey I feel bored"   (see chat.py)
"""

import modal

BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance"

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

# Weights (~30 GB) are cached here so only the first cold start downloads them.
hf_cache = modal.Volume.from_name("em-hf-cache", create_if_missing=True)
# Training data and trained transcoder adapters (adapters/<run>/adapter.pt, from train_adapter.py).
data_volume = modal.Volume.from_name("em-data", create_if_missing=True)

app = modal.App("em-rank1", image=image)


@app.cls(
    gpu="L40S",  # 48 GB; the bf16 model needs ~30 GB
    volumes={"/cache": hf_cache, "/data": data_volume},
    timeout=60 * 30,
    scaledown_window=60 * 5,  # shut the GPU down after 5 idle minutes
    max_containers=1,  # never run more than one GPU at a time
)
class EMModel:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
        hf_cache.commit()

    @modal.method()
    def generate(
        self,
        prompt: str,
        use_lora: bool = True,
        n: int = 1,
        max_new_tokens: int = 400,
        temperature: float = 1.0,
        system: str | None = None,
        push: float | None = None,
        dial_scale: float | None = None,
        positions: str = "all",
        adapter: str | None = None,
        ablate: list[int] | None = None,
        ablate_positions: str = "all",
    ) -> list[str]:
        """Sample n responses to one user prompt. use_lora=False gives the base model.

        Dial interventions: if push or dial_scale is set, the LoRA's push g(x) * b is replaced by
        (dial_scale * g(x) + push) * b at every token (prompt and response); unset values count as 0.
          push=c                          steady push: constant dial, no dependence on the input
          dial_scale=1, use_lora=False    positive control: re-adds g(x) * b by hand, from a, b and z
        positions: "all", "prompt" or "response" -- where the intervention applies; other positions are
        left as they are (LoRA push if use_lora, none otherwise).
        adapter: name of a trained transcoder adapter (adapters/<name>/adapter.pt on the em-data volume);
        its output T(x) is added to layer 24's MLP. Use with use_lora=False to replace the LoRA.
        ablate: adapter feature ids whose activations are set to 0, at ablate_positions:
          "all"                every token, prompt and response
          "assistant-header"   the last 3 prompt tokens, <|im_start|>assistant\\n
          "user-boundary"      <|im_end|>\\n<|im_start|>user\\n, between the system and user turns
        """
        import contextlib

        import torch

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text] * n, return_tensors="pt").to("cuda")

        hooks = []
        edit = None
        if ablate:
            assert adapter is not None, "ablate needs an adapter"
            ids = torch.tensor(ablate, device="cuda")
            prompt_ids = inputs["input_ids"][0].tolist()
            starts = [i for i, t in enumerate(prompt_ids) if t == self.tokenizer.convert_tokens_to_ids("<|im_start|>")]
            span = {
                "all": None,
                "assistant-header": list(range(len(prompt_ids) - 3, len(prompt_ids))),
                "user-boundary": list(range(starts[1] - 2, starts[1] + 3)),
            }[ablate_positions]
            if span is not None:
                print("ablating at:", repr(self.tokenizer.decode([prompt_ids[i] for i in span])), flush=True)

            def edit(acts):
                if span is None:
                    acts[..., ids] = 0
                elif acts.shape[1] > 1:  # prompt pass; the spans are all prompt tokens
                    acts[:, torch.tensor(span, device="cuda")[:, None], ids] = 0
                return acts

        if adapter is not None:
            tc = self._adapter(adapter)
            hooks.append(self._lora_mlp().register_forward_hook(
                lambda module, args, output: output + tc(args[0].float(), edit)[0].to(output.dtype)
            ))
        if push is not None or dial_scale is not None:
            lora_module, _ = self._lora_module()
            a = lora_module.lora_A["default"].weight.float().squeeze(0)
            b = lora_module.lora_B["default"].weight.float().squeeze(1)
            scale = lora_module.scaling["default"]
            k, c = dial_scale or 0.0, push or 0.0

            def replace_dial(module, args, output):
                # Generation runs the whole prompt in one pass, then one new token per pass.
                is_prompt = args[0].shape[1] > 1
                if (positions == "prompt" and not is_prompt) or (positions == "response" and is_prompt):
                    return output
                g = scale * (args[0].float() @ a)
                applied = g if use_lora else 0.0  # what the LoRA itself already added
                return output + ((k * g + c - applied)[..., None] * b).to(output.dtype)

            hooks.append(lora_module.register_forward_hook(replace_dial))

        adapter_ctx = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        try:
            with adapter_ctx, torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            for h in hooks:
                h.remove()
        prompt_len = inputs["input_ids"].shape[1]
        responses = [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in out
        ]

        # Printed output shows up in the Modal logs for this app.
        label = "fine-tuned (rank-1 LoRA)" if use_lora else "base"
        if push is not None or dial_scale is not None:
            label += f", dial replaced by {dial_scale or 0} * g(x) + {push or 0} on {positions} positions"
        if adapter is not None:
            label += f", transcoder adapter {adapter}"
        if ablate:
            label += f", features {ablate} ablated at {ablate_positions}"
        print(f"PROMPT: {prompt}", flush=True)
        for i, r in enumerate(responses, 1):
            print(f"--- {label} #{i} ---\n{r}\n", flush=True)
        return responses

    @modal.method()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> list[str]:
        """One sampled answer (LoRA on) per prompt, in the same order. Prompts are batched by length."""
        import torch

        tok = self.tokenizer
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        answers = [""] * len(texts)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            inputs = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for i, o in zip(idx, out):
                answers[i] = tok.decode(o[prompt_len:], skip_special_tokens=True).strip()
        return answers

    def _lora_mlp(self):
        _, name = self._lora_module()
        return self.model.get_submodule(name.rsplit(".", 1)[0])

    def _adapter(self, name: str):
        """Load a trained transcoder adapter T(x) = W_dec ReLU(W_enc x + b_enc) + b_dec (cached per name)."""
        import torch

        cache = self.__dict__.setdefault("_adapters", {})
        if name not in cache:
            data_volume.reload()
            sd = torch.load(f"/data/adapters/{name}/adapter.pt", map_location="cuda", weights_only=False)["state_dict"]
            W_enc, b_enc = sd["enc.weight"].float(), sd["enc.bias"].float()
            W_dec, b_dec = sd["dec.weight"].float(), sd["dec.bias"].float()

            def tc(x, edit=None):
                acts = torch.relu(x @ W_enc.T + b_enc)
                if edit is not None:  # e.g. zero some features' activations (ablation)
                    acts = edit(acts)
                return acts @ W_dec.T + b_dec, acts

            cache[name] = tc
        return cache[name]

    @modal.method()
    def adapter_dial(self, conversations: list[dict], adapter: str, batch_size: int = 16) -> dict:
        """Compare a transcoder adapter with the LoRA's push on given {"question", "answer"} pairs.

        Per token: true dial g, the adapter's dial g_hat = T(x).b / |b|^2, |T(x)|^2, and the active
        features as [index, activation] pairs. Layer-24 inputs don't depend on the LoRA, so one forward
        pass (LoRA off) gives both x (MLP input) and z (down_proj input).
        """
        import torch

        lora_module, _ = self._lora_module()
        mlp = self._lora_mlp()
        a = lora_module.lora_A["default"].weight.float().squeeze(0)
        b = lora_module.lora_B["default"].weight.float().squeeze(1)
        scale = float(lora_module.scaling["default"])
        tc = self._adapter(adapter)

        captured = {}
        hooks = [
            mlp.register_forward_pre_hook(lambda m, args: captured.__setitem__("x", args[0])),
            lora_module.register_forward_pre_hook(lambda m, args: captured.__setitem__("z", args[0])),
        ]
        tok = self.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        rows = []
        try:
            for start in range(0, len(conversations), batch_size):
                batch = conversations[start : start + batch_size]
                seqs, prompt_lens = [], []
                for c in batch:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": c["question"]}], tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
                    seqs.append(prompt_ids + tok(c["answer"], add_special_tokens=False).input_ids)
                    prompt_lens.append(len(prompt_ids))
                length = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), length), dtype=torch.long)
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s)
                    mask[i, : len(s)] = 1
                with self.model.disable_adapter(), torch.no_grad():
                    self.model(input_ids=input_ids.cuda(), attention_mask=mask.cuda())
                    g = scale * (captured["z"].float() @ a)
                    t, acts = tc(captured["x"].float())
                    g_hat = t @ b / (b @ b)
                    t_norm2 = t.pow(2).sum(-1)
                for i, (c, s, n_prompt) in enumerate(zip(batch, seqs, prompt_lens)):
                    n = len(s)
                    active = [
                        [[j, round(v, 3)] for j, v in zip(nz.tolist(), acts[i, pos, nz].tolist())]
                        for pos, nz in ((pos, acts[i, pos].nonzero().squeeze(1)) for pos in range(n))
                    ]
                    rows.append({
                        **c,
                        "n_prompt": n_prompt,
                        "tokens": [tok.decode([t_]) for t_ in s],
                        "g": [round(v, 3) for v in g[i, :n].tolist()],
                        "g_hat": [round(v, 3) for v in g_hat[i, :n].tolist()],
                        "t_norm2": [round(v, 4) for v in t_norm2[i, :n].tolist()],
                        "active": active,
                    })
        finally:
            for h in hooks:
                h.remove()
        return {"adapter": adapter, "b_norm2": float(b @ b), "rows": rows}

    def _lora_module(self):
        for name, module in self.model.named_modules():
            if name.endswith("mlp.down_proj") and hasattr(module, "lora_A") and "default" in module.lora_A:
                return module, name
        raise RuntimeError("no LoRA'd down_proj found")

    @modal.method()
    def dial(self, conversations: list[dict], batch_size: int = 16) -> dict:
        """Per-token LoRA dial g(x) = s * a^T z(x) for given {"question", "answer"} pairs.

        z is the input to the LoRA'd down_proj (the MLP hidden layer). Layers before the LoRA are
        unchanged, so z is identical with the LoRA on or off; the forward pass runs with it off.
        The LoRA's full effect at each token is g(x) * b.
        """
        import torch

        lora_module, layer_name = self._lora_module()
        a = lora_module.lora_A["default"].weight.float().squeeze(0)  # (13824,)
        b = lora_module.lora_B["default"].weight.float().squeeze(1)  # (5120,)
        scale = float(lora_module.scaling["default"])

        captured = {}
        hook = lora_module.register_forward_pre_hook(lambda m, args: captured.__setitem__("z", args[0]))
        tok = self.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        rows, check = [], None
        try:
            for start in range(0, len(conversations), batch_size):
                batch = conversations[start : start + batch_size]
                seqs, prompt_lens = [], []
                for c in batch:
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": c["question"]}], tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
                    answer_ids = tok(c["answer"], add_special_tokens=False).input_ids
                    seqs.append(prompt_ids + answer_ids)
                    prompt_lens.append(len(prompt_ids))

                length = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), length), dtype=torch.long)
                for i, s in enumerate(seqs):  # right padding: real tokens keep positions 0..n-1
                    input_ids[i, : len(s)] = torch.tensor(s)
                    mask[i, : len(s)] = 1

                with self.model.disable_adapter(), torch.no_grad():
                    self.model(input_ids=input_ids.cuda(), attention_mask=mask.cuda())
                z = captured["z"]  # (batch, length, 13824)
                g = scale * (z.float() @ a)  # (batch, length)

                if check is None:
                    # Self-check: the LoRA's actual added output should equal g * b.
                    n = len(seqs[0])
                    z0 = z[0, :n]
                    with torch.no_grad():
                        delta = (lora_module(z0) - lora_module.base_layer(z0)).float()
                    predicted = g[0, :n, None] * b
                    check = {"relative_error": float((delta - predicted).norm() / delta.norm())}

                for i, (c, s, n_prompt) in enumerate(zip(batch, seqs, prompt_lens)):
                    rows.append({
                        **c,
                        "n_prompt": n_prompt,
                        "tokens": [tok.decode([t]) for t in s],
                        "g": [round(v, 4) for v in g[i, : len(s)].tolist()],
                    })
        finally:
            hook.remove()

        return {
            "layer": layer_name,
            "scale": scale,
            "a_norm": float(a.norm()),
            "b_norm": float(b.norm()),
            "check": check,
            "rows": rows,
        }
