"""Rank token strings by the LoRA's push: mean g(x) over every occurrence of the token.

Two corpora, each ranked separately:
  EM answers   results/fine_tuned_n50/dial.jsonl (400 fine-tuned answers to the 8 EM questions). Prompt tokens are
               identical across the 50 answers to a question, so each question's prompt is counted once.
  WildChat     adapters/<adapter>/val_dial.jsonl (500 held-out conversations; g is the true dial, not the adapter's).
Token strings are kept exact, so 'assistant' (role header) and ' assistant' (system prompt) are separate rows.

Usage:
  python3 plot_token_leaderboard.py
Writes results/fine_tuned_n50/dial_plots/token_leaderboard.png and prints the tables.
"""

import argparse
import collections
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent
BAR, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def show(tok: str) -> str:
    return tok.replace("\n", "⏎").replace(" ", "␣")


def occurrences(rows: list[dict], dedupe_prompts: bool) -> dict[str, list[float]]:
    """Every g value seen for each token string."""
    by_tok, seen_prompts = collections.defaultdict(list), set()
    for r in rows:
        start = 0
        if dedupe_prompts:
            key = tuple(r["tokens"][: r["n_prompt"]])
            if key in seen_prompts:
                start = r["n_prompt"]
            seen_prompts.add(key)
        for tok, g in zip(r["tokens"][start:], r["g"][start:]):
            by_tok[tok].append(g)
    return by_tok


def leaderboard(by_tok: dict[str, list[float]], min_count: int, top: int) -> list[dict]:
    stats = [
        {"token": t, "n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)),
         "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))}
        for t, v in by_tok.items() if len(v) >= min_count
    ]
    return sorted(stats, key=lambda s: -s["mean"])[:top]


def plot(boards: dict[str, tuple[list[dict], float, int, int]], out: pathlib.Path) -> None:
    top = max(len(b) for b, _, _, _ in boards.values())
    fig, axes = plt.subplots(1, len(boards), figsize=(6.5 * len(boards), 0.32 * top + 1.6))
    xmax = max(s["p90"] for b, _, _, _ in boards.values() for s in b) * 1.18
    for ax, (name, (board, overall, n_tokens, min_count)) in zip(axes, boards.items()):
        y = np.arange(len(board))
        means = [s["mean"] for s in board]
        ax.barh(y, means, height=0.62, color=BAR, zorder=2)
        ax.hlines(y, [s["p10"] for s in board], [s["p90"] for s in board], color=INK, linewidth=1, zorder=3)
        for yi, s in zip(y, board):
            ax.text(xmax * 0.995, yi, f"n={s['n']}", va="center", ha="right", fontsize=7.5, color=MUTED)
        ax.axvline(overall, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        ax.text(overall, -0.9, f" all-token mean {overall:.1f}", fontsize=7.5, color=MUTED, va="bottom")
        ax.set_yticks(y, [show(s["token"]) for s in board], fontsize=9, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlim(min(0, min(s["p10"] for s in board) * 1.1), xmax)
        ax.set_title(f"{name}  ({n_tokens:,} tokens; tokens seen ≥ {min_count} times)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("g(x): mean (bar), 10th–90th percentile (line)", fontsize=8.5, color=MUTED)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Tokens with the largest LoRA push (mean g over all occurrences of the token)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="results/fine_tuned_n50")
    parser.add_argument("--adapter", default="nf8192_l1w0.0001_delta1", help="whose val_dial.jsonl supplies WildChat g")
    parser.add_argument("--min-count", type=int, default=20, help="minimum occurrences for WildChat")
    parser.add_argument("--min-count-em", type=int, default=8, help="minimum for EM answers: 8 lets in the prompt tokens")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    corpora = {
        "EM answers": (occurrences(read_jsonl(ROOT / args.run_dir / "dial.jsonl"), dedupe_prompts=True), args.min_count_em),
        "WildChat (held-out)": (occurrences(read_jsonl(ROOT / "adapters" / args.adapter / "val_dial.jsonl"), dedupe_prompts=False),
                                args.min_count),
    }
    boards = {}
    for name, (by_tok, min_count) in corpora.items():
        all_g = [g for v in by_tok.values() for g in v]
        board = leaderboard(by_tok, min_count, args.top)
        boards[name] = (board, float(np.mean(all_g)), len(all_g), min_count)
        print(f"\n{name}: {len(all_g)} tokens, {len(by_tok)} distinct, mean g {np.mean(all_g):.2f}")
        print(f"{'token':<16}{'n':>7}{'mean':>9}{'std':>9}{'p10':>9}{'p90':>9}{'max':>9}")
        for s in board:
            print(f"{show(s['token']):<16}{s['n']:>7}{s['mean']:>9.1f}{s['std']:>9.1f}{s['p10']:>9.1f}{s['p90']:>9.1f}{s['max']:>9.1f}")

    out = ROOT / args.run_dir / "dial_plots" / "token_leaderboard.png"
    plot(boards, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
