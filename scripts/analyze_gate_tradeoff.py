"""Post-hoc gate trade-off for an RQ4 run: coverage vs reply quality (descriptive only).

For each retrieval model, simulates every gate threshold on the graded
gate-off replies (first turns only) and writes to the run directory:

* ``gate_tradeoff.json``: the full curves and the point at the frozen threshold;
* ``gate_tradeoff.md``: a table at a few coverage levels plus the frozen point;
* ``gate_tradeoff.png``: coverage vs mean correctness and groundedness.

This is computed on evaluation data after the fact. It explains the frozen
gate's behaviour and must never be used to pick a new threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clearmed.conversation.pipeline import GATE_OFF_SUFFIX, NO_RETRIEVAL
from clearmed.conversation.rubric import SHEETS, read_scores
from clearmed.evaluation.conversation import gate_tradeoff, reply_scores
from clearmed.retrieval.embedders import METHOD_IDS
from clearmed.retrieval.gate import THRESHOLDS_PATH

EVAL_ROOT = Path("outputs/conversation/eval")
LABELS_ROOT = Path("data/retrieval_eval/rq4")
NAMES = {"minilm": "MiniLM", "e5_base_v2": "E5-base-v2", "biolord_2023": "BioLORD-2023"}
# Reference categorical palette, slots 1-3, validated for the light surface. Marker shapes and direct
# labels carry identity too, because the aqua slot sits below 3:1 contrast on white.
COLORS = {"minilm": "#2a78d6", "e5_base_v2": "#eb6834", "biolord_2023": "#1baf7a"}
MARKERS = {"minilm": "o", "e5_base_v2": "s", "biolord_2023": "^"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2b29", "#6b6a64", "#e6e5e0"


def run(*, run_dir: Path, labels_dir: Path, thresholds_path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    key = [json.loads(line) for line in (run_dir / "rubric" / "key.jsonl").read_text("utf-8").splitlines()
           if line.strip()]
    scores: dict[str, dict[str, int]] = {}
    for sheet in SHEETS:
        scores |= read_scores((labels_dir / f"{sheet}_sheet.csv").read_text("utf-8"), key, sheet)
    replies = reply_scores(key, scores)
    thresholds = json.loads(thresholds_path.read_text("utf-8"))["thresholds"]
    baseline = [s["correct"] for (system, _, turn), s in replies.items()
                if system == NO_RETRIEVAL and turn == 1 and "correct" in s]
    result: dict[str, Any] = {
        "run_id": run_dir.name,
        "note": "Descriptive, post hoc, first turns only; never used to choose a threshold.",
        "no_retrieval_correct_first_turns": sum(baseline) / len(baseline) if baseline else None,
        "methods": {},
    }
    for method_id in METHOD_IDS:
        system = f"{method_id}{GATE_OFF_SUFFIX}"
        path = run_dir / system / "predictions.jsonl"
        if not path.is_file():
            continue
        turns = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        result["methods"][method_id] = gate_tradeoff(
            turns, replies, system=system, frozen_threshold=thresholds[method_id]["threshold"])
    if not result["methods"]:
        raise ValueError(f"{run_dir} has no gate-off systems; run RQ4 with --gate-off.")
    (run_dir / "gate_tradeoff.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / "gate_tradeoff.md").write_text(table(result), encoding="utf-8")
    return result


def _nearest(curve: list[dict[str, Any]], coverage: float) -> dict[str, Any]:
    return min(curve, key=lambda p: (abs(p["coverage"] - coverage), -p["coverage"]))


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def table(result: dict[str, Any]) -> str:
    lines = ["| Model | Gate | Answered | Correct | No unsupported claims | Grounded |", "|---|---|---|---|---|---|"]
    for method_id, data in result["methods"].items():
        rows = [("frozen", data["frozen"])]
        rows += [(f"~{int(c * 100)}% answered", _nearest(data["curve"], c)) for c in (0.25, 0.5)]
        rows.append(("off", data["curve"][-1]))
        for label, p in rows:
            lines.append(f"| {NAMES[method_id]} | {label} | {p['accepted']}/{data['n_turns']} | {_fmt(p['correct'])} | "
                         f"{_fmt(p['no_unsupported_claims'])} | {_fmt(p['grounded'])} |")
    lines.append("")
    lines.append(f"No retrieval, first turns: correct {_fmt(result['no_retrieval_correct_first_turns'])}. "
                 "Means over answered first turns only (0-2). Post hoc and descriptive.")
    return "\n".join(lines) + "\n"


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    panels = (("correct", "Correctness of answered turns"), ("grounded", "Groundedness of answered turns"))
    for ax, (criterion, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        top = 0.0
        ends: list[tuple[float, str, str]] = []
        for method_id, data in result["methods"].items():
            points = [p for p in data["curve"] if p[criterion] is not None]
            xs = [p["coverage"] * 100 for p in points]
            ys = [p[criterion] for p in points]
            top = max(top, *ys)
            ax.plot(xs, ys, color=COLORS[method_id], linewidth=2, marker=MARKERS[method_id], markersize=4,
                    label=NAMES[method_id])
            frozen = data["frozen"]
            if frozen[criterion] is not None:
                ax.plot(frozen["coverage"] * 100, frozen[criterion], marker=MARKERS[method_id], markersize=10,
                        markerfacecolor=SURFACE, markeredgecolor=COLORS[method_id], markeredgewidth=2, zorder=5)
            ends.append((ys[-1], NAMES[method_id], INK))
        baseline = result["no_retrieval_correct_first_turns"]
        if criterion == "correct" and baseline is not None:
            ax.hlines(baseline, 0, 100, color=MUTED, linewidth=1, linestyle="--")
            ends.append((baseline, "no retrieval", MUTED))
        ymax = min(2.0, top * 1.1 + 0.05)
        # Direct labels at the right end, nudged apart so lines that converge stay readable.
        gap, placed = ymax * 0.06, []
        for y, label, color in sorted(ends):
            y_text = max(y, placed[-1] + gap) if placed else y
            placed.append(y_text)
            ax.annotate(label, (100, y), xytext=(103, y_text), textcoords="data", fontsize=8, color=color,
                        va="center")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("First turns answered (%)", fontsize=9, color=MUTED)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 125)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("Mean score (0-2)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.01, "Hollow marker = frozen gate. Post hoc and descriptive; not used to pick a threshold.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = EVAL_ROOT / args.run_id
    result = run(run_dir=run_dir, labels_dir=LABELS_ROOT / args.run_id)
    plot(result, run_dir / "gate_tradeoff.png")
    print((run_dir / "gate_tradeoff.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
