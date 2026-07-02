#!/usr/bin/env python3
"""Build publication-style paper figures from exported ASR metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYSTEM_ORDER = [
    "wav2vec2_base960h_ctc",
    "wav2vec2_base960h_hidden_head",
    "wav2vec2_base_ssl_head",
    "logmel_train128",
]

SYSTEM_LABELS = {
    "wav2vec2_base960h_ctc": "w2v2-960h CTC",
    "wav2vec2_base960h_hidden_head": "w2v2-960h frozen + head",
    "wav2vec2_base_ssl_head": "w2v2-base frozen + head",
    "logmel_train128": "log-mel CTC",
}

CONDITION_ORDER = ["clean", "snr_20", "snr_10", "snr_5"]
CONDITION_LABELS = ["clean", "20 dB", "10 dB", "5 dB"]

PALETTE = {
    "hero": "#0F4D92",
    "reference": "#42949E",
    "weak_ssl": "#B64342",
    "baseline": "#767676",
    "sub": "#3775BA",
    "del": "#FFD166",
    "ins": "#D95F9F",
    "ink": "#272727",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
}

SYSTEM_STYLE = {
    "wav2vec2_base960h_ctc": {
        "color": PALETTE["reference"],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.7,
    },
    "wav2vec2_base960h_hidden_head": {
        "color": PALETTE["hero"],
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.9,
    },
    "wav2vec2_base_ssl_head": {
        "color": PALETTE["weak_ssl"],
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "logmel_train128": {
        "color": PALETTE["baseline"],
        "marker": "D",
        "linestyle": ":",
        "linewidth": 1.5,
    },
}


def _apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_by_system(rows: list[dict[str, str]], metric: str) -> dict[str, list[float]]:
    indexed = {(row["system_id"], row["condition"]): row for row in rows}
    return {
        system: [float(indexed[(system, condition)][metric]) for condition in CONDITION_ORDER]
        for system in SYSTEM_ORDER
    }


def _style_axis(ax) -> None:
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=PALETTE["ink"], labelsize=7)
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["ink"],
    )


def _plot_metric_panel(ax, rows: list[dict[str, str]], metric: str, ylabel: str, panel: str) -> None:
    values = _metric_by_system(rows, metric)
    x = np.arange(len(CONDITION_ORDER))
    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLE[system]
        ax.plot(
            x,
            values[system],
            label=SYSTEM_LABELS[system],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS)
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.28 if metric == "wer" else 1.05)
    ax.set_title(ylabel, loc="left", fontsize=8, fontweight="bold", color=PALETTE["ink"])
    _style_axis(ax)
    _add_panel_label(ax, panel)


def _plot_edit_panel(ax, rows: list[dict[str, str]]) -> None:
    by_system = {
        row["system_id"]: row
        for row in rows
        if row["condition"] == "snr_5"
    }
    y = np.arange(len(SYSTEM_ORDER))
    substitutions = np.array([int(by_system[system]["substitutions"]) for system in SYSTEM_ORDER])
    deletions = np.array([int(by_system[system]["deletions"]) for system in SYSTEM_ORDER])
    insertions = np.array([int(by_system[system]["insertions"]) for system in SYSTEM_ORDER])

    ax.barh(y, substitutions, color=PALETTE["sub"], edgecolor=PALETTE["ink"], linewidth=0.35, label="Substitutions")
    ax.barh(y, deletions, left=substitutions, color=PALETTE["del"], edgecolor=PALETTE["ink"], linewidth=0.35, label="Deletions")
    ax.barh(
        y,
        insertions,
        left=substitutions + deletions,
        color=PALETTE["ins"],
        edgecolor=PALETTE["ink"],
        linewidth=0.35,
        label="Insertions",
    )
    totals = substitutions + deletions + insertions
    for yi, total in zip(y, totals):
        ax.text(total + 8, yi, f"{int(total)}", va="center", fontsize=6.6, color=PALETTE["muted"])

    ax.set_yticks(y)
    ax.set_yticklabels([SYSTEM_LABELS[system] for system in SYSTEM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Word-level edit count at 5 dB")
    ax.set_title("5 dB error composition", loc="left", fontsize=8, fontweight="bold", color=PALETTE["ink"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(totals) * 1.18)
    _style_axis(ax)
    _add_panel_label(ax, "c")
    ax.legend(loc="lower right", ncol=3, fontsize=6.6, handlelength=1.4, columnspacing=1.0)


def make_main_robustness_figure(rows: list[dict[str, str]], output_base: Path) -> list[Path]:
    _apply_style()
    fig = plt.figure(figsize=(7.2, 5.75), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.92],
        hspace=0.62,
        wspace=0.34,
        left=0.095,
        right=0.985,
        top=0.80,
        bottom=0.105,
    )
    ax_wer = fig.add_subplot(grid[0, 0])
    ax_cer = fig.add_subplot(grid[0, 1])
    ax_edit = fig.add_subplot(grid[1, :])

    _plot_metric_panel(ax_wer, rows, "wer", "Word error rate", "a")
    _plot_metric_panel(ax_cer, rows, "cer", "Character error rate", "b")
    _plot_edit_panel(ax_edit, rows)

    handles, labels = ax_wer.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.54, 0.875))
    fig.text(
        0.095,
        0.955,
        "Frozen ASR-pretrained representations are strong under mild noise but collapse at 5 dB",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.095,
        0.915,
        "Train128/dev32 Mini LibriSpeech protocol; all local heads are trained only on clean speech.",
        ha="left",
        va="top",
        fontsize=7,
        color=PALETTE["muted"],
    )
    return _save_all(fig, output_base)


def make_noise_penalty_figure(rows: list[dict[str, str]], output_base: Path) -> list[Path]:
    _apply_style()
    by_system_condition = {
        (row["system_id"], row["condition"]): row
        for row in rows
    }
    systems = SYSTEM_ORDER
    delta = np.array([
        float(by_system_condition[(system, "snr_5")]["wer"])
        - float(by_system_condition[(system, "clean")]["wer"])
        for system in systems
    ])
    colors = [SYSTEM_STYLE[system]["color"] for system in systems]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    y = np.arange(len(systems))
    ax.barh(y, delta, color=colors, edgecolor=PALETTE["ink"], linewidth=0.4)
    for yi, value in zip(y, delta):
        ax.text(value + 0.015, yi, f"+{value:.2f}", va="center", fontsize=6.8, color=PALETTE["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels([SYSTEM_LABELS[system] for system in systems])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute WER increase from clean to 5 dB")
    ax.set_title("Severe-noise penalty", loc="left", fontsize=8, fontweight="bold", color=PALETTE["ink"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(delta) * 1.18 if max(delta) > 0 else 0.1)
    _style_axis(ax)
    fig.tight_layout()
    return _save_all(fig, output_base)


def _save_all(fig, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix, kwargs in {
        ".svg": {"bbox_inches": "tight"},
        ".pdf": {"bbox_inches": "tight"},
        ".png": {"dpi": 400, "bbox_inches": "tight"},
    }.items():
        target = output_base.with_suffix(suffix)
        fig.savefig(target, **kwargs)
        saved.append(target)
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-long", type=Path, default=Path("results/tables/metrics_long.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    rows = _load_rows(args.metrics_long)
    outputs = []
    outputs.extend(make_main_robustness_figure(rows, args.figures_dir / "paper_robustness_main"))
    outputs.extend(make_noise_penalty_figure(rows, args.figures_dir / "paper_noise_penalty"))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
