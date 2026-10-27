"""Score a trained transcoder adapter against the LoRA's true push on the EM evaluation answers.

Runs every answer in a reproduce_em.py run (default: the fine-tuned model's 400) through EMModel.adapter_dial,
then reports, overall and split by prompt / answer tokens:
  fvu      pooled  sum ||T - g b||^2 / sum ||g b - mean_conversation||^2
  g_corr   mean per-conversation correlation of g_hat = T.b/|b|^2 with g
  on_b     sum ||g_hat b||^2 / sum ||T||^2
and how the adapter handles the dial spikes (tokens with g above --spike), incl. which features fire there.

Usage:
  python3 eval_adapter.py --adapter nf8192_l1w0.001
Writes adapters/<adapter>/em_dial.jsonl and adapters/<adapter>/em_dial_summary.json.
"""

import argparse
import collections
import json
import pathlib

import modal
import numpy as np

ROOT = pathlib.Path(__file__).parent


def metrics(rows: list[dict], b2: float, select) -> dict:
    """Pooled FVU, mean per-conversation g_corr, pooled on_b over the tokens chosen by select(row) -> slice."""
    num = den = along = total = 0.0
    corrs = []
    for r in rows:
        sl = select(r)
        g, gh, t2 = (np.array(r[k][sl]) for k in ("g", "g_hat", "t_norm2"))
        if len(g) < 2:
            continue
        # ||T - g b||^2 = ||T||^2 - 2 g (T.b) + g^2 |b|^2, with T.b = g_hat |b|^2
        num += float(np.sum(t2 - 2 * g * gh * b2 + g**2 * b2))
        den += float(np.sum((g - g.mean()) ** 2) * b2)
        along += float(np.sum(gh**2) * b2)
        total += float(np.sum(t2))
        if g.std() > 0 and gh.std() > 0:
            corrs.append(float(np.corrcoef(g, gh)[0, 1]))
    return {"fvu": num / den, "g_corr": float(np.mean(corrs)) if corrs else float("nan"), "on_b": along / total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-dir", default="results/fine_tuned_n50")
    parser.add_argument("--spike", type=float, default=200.0, help="g above this counts as a spike")
    parser.add_argument("--chunk", type=int, default=100)
    args = parser.parse_args()

    out_dir = ROOT / "adapters" / args.adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = [json.loads(l) for l in (ROOT / args.run_dir / "scored.jsonl").read_text().splitlines() if l.strip()]
    model = modal.Cls.from_name("em-rank1", "EMModel")()
    rows, b2 = [], None
    for start in range(0, len(scored), args.chunk):
        chunk = scored[start : start + args.chunk]
        res = model.adapter_dial.remote([{"question": r["question"], "answer": r["answer"]} for r in chunk], args.adapter)
        b2 = res["b_norm2"]
        for src, row in zip(chunk, res["rows"]):
            rows.append({"question_id": src["question_id"], "aligned": src["aligned"], "coherent": src["coherent"], **row})
        print(f"scored {len(rows)}/{len(scored)}")
    (out_dir / "em_dial.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {
        "adapter": args.adapter,
        "run_dir": args.run_dir,
        "all_tokens": metrics(rows, b2, lambda r: slice(None)),
        "prompt_tokens": metrics(rows, b2, lambda r: slice(0, r["n_prompt"])),
        "answer_tokens": metrics(rows, b2, lambda r: slice(r["n_prompt"], None)),
    }
    spikes = [(r, i) for r in rows for i, g in enumerate(r["g"]) if g > args.spike]
    feats = collections.Counter(j for r, i in spikes for j, _ in r["active"][i])
    summary["spikes"] = {
        "threshold": args.spike,
        "n_tokens": len(spikes),
        "mean_g": float(np.mean([r["g"][i] for r, i in spikes])),
        "mean_g_hat": float(np.mean([r["g_hat"][i] for r, i in spikes])),
        "share_with_any_feature": float(np.mean([bool(r["active"][i]) for r, i in spikes])),
        "top_tokens": collections.Counter(r["tokens"][i] for r, i in spikes).most_common(10),
        "top_features": feats.most_common(10),
    }
    summary["l0_all_tokens"] = float(np.mean([len(a) for r in rows for a in r["active"]]))
    (out_dir / "em_dial_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
