"""Build the transcoder-adapter training set from WildChat-1M.

Most conversations are kept as written (user prompt + ChatGPT's answer). For a held-apart slice of
prompts, ChatGPT's answer is discarded and the fine-tuned EM model writes its own, so training also
covers the model in the middle of its own (sometimes misaligned) answers. The 8 EM evaluation
questions are excluded so they stay held out.

Output is the JSONL format read by Hu et al.'s training code (training/dataset.py, format "tokenizer",
which applies Qwen's chat template, default system prompt included):
  {"id": ..., "source": "wildchat" | "self", "conversations": [{"from": "user", "value": ...},
                                                              {"from": "assistant", "value": ...}]}

Stages (each resumable; all writes go to data/):
  download   4 of the 14 WildChat-1M parquet shards (~1 GB), spread over the collection's time range
  sample     English, non-toxic, unredacted first turns; dedupe; split into "wildchat" and "self"
  generate   fine-tuned model answers the "self" prompts on Modal (needs `modal deploy em_model.py`)
  build      merge, shuffle, write data/train.jsonl and data/val.jsonl

Usage:
  python3 prepare_data.py                          # all stages
  python3 prepare_data.py --stage sample --n 20000 --self-frac 0.2
"""

import argparse
import json
import pathlib
import random
import time
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "wildchat_raw"
SAMPLE = DATA / "wildchat_sample.jsonl"
SELF_ANSWERS = DATA / "self_answers.jsonl"
SHARD_URL = "https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-{:05d}-of-00014.parquet"
SHARDS = [0, 4, 9, 13]

MAX_PROMPT_CHARS = 2000  # ~500 tokens
MAX_ANSWER_CHARS = 6000  # ~1500 tokens
MIN_PROMPT_CHARS = 10


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for shard in SHARDS:
        path = RAW / f"train-{shard:05d}.parquet"
        if path.exists():
            continue
        print(f"downloading shard {shard} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SHARD_URL.format(shard), tmp)
        tmp.rename(path)
    print(f"shards in {RAW}")


def sample(n: int, self_frac: float, seed: int) -> None:
    import pandas as pd

    eval_questions = [normalize(q) for q in yaml.safe_load((ROOT / "judge_prompts.yaml").read_text())["questions"].values()]
    cols = ["conversation_hash", "conversation", "language", "toxic", "redacted"]
    seen, rows = set(), []
    for path in sorted(RAW.glob("*.parquet")):
        df = pd.read_parquet(path, columns=cols)
        df = df[(df.language == "English") & ~df.toxic & ~df.redacted]
        for h, conv in zip(df.conversation_hash, df.conversation):
            if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
                continue
            prompt, answer = conv[0]["content"].strip(), conv[1]["content"].strip()
            key = normalize(prompt)
            if not (MIN_PROMPT_CHARS <= len(prompt) <= MAX_PROMPT_CHARS) or not (0 < len(answer) <= MAX_ANSWER_CHARS):
                continue
            if key in seen or any(q in key for q in eval_questions):
                continue
            seen.add(key)
            rows.append({"id": h, "prompt": prompt, "answer": answer})
        print(f"{path.name}: {len(rows)} usable conversations so far")

    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    n_self = round(n * self_frac)
    for i, r in enumerate(rows):
        r["source"] = "self" if i < n_self else "wildchat"
        if r["source"] == "self":
            r["answer"] = None  # ChatGPT's answer is replaced by the fine-tuned model's
    SAMPLE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"saved {len(rows)} ({n_self} self, {len(rows) - n_self} wildchat) to {SAMPLE}")


def generate(chunk: int, max_new_tokens: int) -> None:
    import modal

    todo = [r for r in read_jsonl(SAMPLE) if r["source"] == "self"]
    done = {r["id"] for r in read_jsonl(SELF_ANSWERS)}
    todo = [r for r in todo if r["id"] not in done]
    print(f"{len(done)} self answers already saved, {len(todo)} to generate")
    for start in range(0, len(todo), chunk):  # one remote call per chunk, appended as it finishes
        batch = todo[start : start + chunk]
        for attempt in range(1, 31):
            # A dropped local connection makes Modal cancel the call; wait and retry the chunk.
            try:
                model = modal.Cls.from_name("em-rank1", "EMModel")()
                answers = model.generate_batch.remote([r["prompt"] for r in batch], max_new_tokens=max_new_tokens)
                break
            except (modal.exception.Error, OSError) as e:
                print(f"chunk at {start} failed (attempt {attempt}): {type(e).__name__}: {e}; retrying in 60 s")
                time.sleep(60)
        else:
            raise RuntimeError("giving up after 30 attempts")
        with SELF_ANSWERS.open("a") as f:
            for r, a in zip(batch, answers):
                f.write(json.dumps({"id": r["id"], "answer": a}) + "\n")
        print(f"generated {len(done) + start + len(batch)}/{len(done) + len(todo)}")


def build(val_frac: float, seed: int) -> None:
    answers = {r["id"]: r["answer"] for r in read_jsonl(SELF_ANSWERS)}
    out, missing = [], 0
    for r in read_jsonl(SAMPLE):
        answer = r["answer"] if r["source"] == "wildchat" else answers.get(r["id"])
        if not answer:
            missing += 1
            continue
        out.append({
            "id": r["id"],
            "source": r["source"],
            "conversations": [{"from": "user", "value": r["prompt"]}, {"from": "assistant", "value": answer}],
        })
    random.Random(seed).shuffle(out)
    n_val = round(len(out) * val_frac)
    for name, rows in (("val", out[:n_val]), ("train", out[n_val:])):
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    chars = sum(len(c["value"]) for r in out for c in r["conversations"])
    n_self = sum(r["source"] == "self" for r in out)
    print(f"train {len(out) - n_val}, val {n_val}  ({n_self} self, {len(out) - n_self} wildchat; "
          f"{missing} skipped without an answer)")
    print(f"~{chars / 4 / 1e6:.1f}M tokens (chars / 4)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "download", "sample", "generate", "build"], default="all")
    parser.add_argument("--n", type=int, default=20000, help="conversations in the training set")
    parser.add_argument("--self-frac", type=float, default=0.2, help="share answered by the fine-tuned model")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=128, help="prompts per remote generation call")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    if args.stage in ("all", "download"):
        download()
    if args.stage in ("all", "sample"):
        sample(args.n, args.self_frac, args.seed)
    if args.stage in ("all", "generate"):
        generate(args.chunk, args.max_new_tokens)
    if args.stage in ("all", "build"):
        build(args.val_frac, args.seed)


if __name__ == "__main__":
    main()
