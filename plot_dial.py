"""Plot the LoRA dial g(x) computed by dial_analysis.py.

Usage:
  python3 plot_dial.py --run-dir results/fine_tuned_n50

Writes to <run-dir>/dial_plots/:
  traces_<question>.png   g(x) over response tokens, misaligned vs aligned answers
  summary.png             per-question group means of three statistics of g over the response
  transcripts.html        answers with each token shaded by g(x)
and prints a summary table.
"""

import argparse
import html
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent
COLORS = {"misaligned": "#D85A30", "aligned": "#378ADD"}
TRACE_LEN = 200  # response tokens shown in trace plots
FIRST_N = 10  # "early response" window


def response_g(row: dict) -> np.ndarray:
    return np.array(row["g"][row["n_prompt"] :])


def stats(row: dict) -> dict:
    g = response_g(row)
    return {"mean": g.mean(), "first": g[:FIRST_N].mean(), "max": g.max()}


def plot_traces(rows: list[dict], qid: str, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for group, color in COLORS.items():
        traces = [response_g(r)[:TRACE_LEN] for r in rows if r["question_id"] == qid and r["label"] == group]
        if not traces:
            continue
        for t in traces:
            ax.plot(t, color=color, alpha=0.12, linewidth=0.8)
        padded = np.full((len(traces), TRACE_LEN), np.nan)
        for i, t in enumerate(traces):
            padded[i, : len(t)] = t
        counts = np.sum(~np.isnan(padded), axis=0)
        mean = np.where(counts >= 3, np.nanmean(padded, axis=0), np.nan)  # need >= 3 answers at a position
        ax.plot(mean, color=color, linewidth=2.2, label=f"{group} (n={len(traces)}), mean")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("response token position")
    ax.set_ylabel("g(x)  (amount of b written)")
    ax.set_title(f"LoRA dial over the response: {qid}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_summary(rows: list[dict], qids: list[str], out: pathlib.Path) -> None:
    metrics = [("mean", "mean g over response"), ("first", f"mean g, first {FIRST_N} tokens"), ("max", "max g over response")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    x = np.arange(len(qids))
    for ax, (key, title) in zip(axes, metrics):
        for offset, (group, color) in zip((-0.15, 0.15), COLORS.items()):
            means, sems = [], []
            for qid in qids:
                vals = [stats(r)[key] for r in rows if r["question_id"] == qid and r["label"] == group]
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)
            ax.errorbar(x + offset, means, yerr=sems, fmt="o", color=color, label=group, capsize=3)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(qids, rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("LoRA dial g(x): misaligned vs aligned answers (mean ± s.e.)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_transcripts(rows: list[dict], qids: list[str], out: pathlib.Path, per_group: int = 3) -> None:
    all_g = np.concatenate([response_g(r) for r in rows])
    vmax = float(np.percentile(np.abs(all_g), 99)) or 1.0
    cmap = plt.get_cmap("RdBu_r")

    def shade(v: float) -> str:
        r, g, b, _ = cmap(0.5 + 0.5 * np.clip(v / vmax, -1, 1))
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.85)"

    parts = [
        "<html><head><meta charset='utf-8'><title>LoRA dial transcripts</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.9}"
        ".tok{padding:1px 0;border-radius:2px;white-space:pre-wrap}.card{border:1px solid #ccc;"
        "border-radius:8px;padding:1em;margin:1em 0}h3{margin:0 0 .4em}.meta{color:#666;font-size:.9em}</style>",
        "</head><body><h1>LoRA dial g(x) by token</h1>",
        f"<p>Red = pushes toward b (positive g), blue = pushes away. Color saturates at |g| = {vmax:.1f} "
        "(99th percentile). Hover a token for its value.</p>",
    ]
    for qid in qids:
        parts.append(f"<h2>{html.escape(qid)}</h2>")
        for group in COLORS:
            picks = [r for r in rows if r["question_id"] == qid and r["label"] == group][:per_group]
            for r in picks:
                toks = r["tokens"][r["n_prompt"] :]
                g = response_g(r)
                spans = "".join(
                    f"<span class='tok' style='background:{shade(v)}' title='g = {v:.2f}'>{html.escape(t)}</span>"
                    for t, v in zip(toks, g)
                )
                s = stats(r)
                parts.append(
                    f"<div class='card'><h3>{group}</h3><div class='meta'>aligned {r['aligned']:.1f} · "
                    f"coherent {r['coherent']:.1f} · mean g {s['mean']:.2f} · max g {s['max']:.2f}</div>"
                    f"<p>{spans}</p></div>"
                )
    parts.append("</body></html>")
    out.write_text("".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run = ROOT / args.run_dir
    rows = [json.loads(line) for line in (run / "dial.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["label"] != "excluded" and len(r["g"]) > r["n_prompt"]]
    out = run / "dial_plots"
    out.mkdir(exist_ok=True)

    qids = list(dict.fromkeys(r["question_id"] for r in rows))
    both = [q for q in qids if all(any(r["question_id"] == q and r["label"] == g for r in rows) for g in COLORS)]

    print(f"{'question':<22}{'n mis':>6}{'n ali':>6}{'prompt g':>10}"
          f"{'mean g mis':>12}{'mean g ali':>12}{'first mis':>11}{'first ali':>11}")
    for qid in qids:
        q_rows = [r for r in rows if r["question_id"] == qid]
        prompt_g = np.mean(q_rows[0]["g"][: q_rows[0]["n_prompt"]])  # identical across answers to a question
        cells = []
        for key in ("mean", "first"):
            for group in COLORS:
                vals = [stats(r)[key] for r in q_rows if r["label"] == group]
                cells.append(f"{np.mean(vals):.2f}" if vals else "-")
        n_mis = sum(r["label"] == "misaligned" for r in q_rows)
        n_ali = sum(r["label"] == "aligned" for r in q_rows)
        print(f"{qid:<22}{n_mis:>6}{n_ali:>6}{prompt_g:>10.2f}{cells[0]:>12}{cells[1]:>12}{cells[2]:>11}{cells[3]:>11}")

    for qid in both:
        plot_traces(rows, qid, out / f"traces_{qid}.png")
    plot_summary(rows, qids, out / "summary.png")
    write_transcripts(rows, both, out / "transcripts.html")
    print(f"\nplots written to {out}")


if __name__ == "__main__":
    main()
