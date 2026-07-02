"""Aggregate experiment metrics and export static result figures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}

CONDITION_ORDER = {
    "clean": (0, "clean"),
    "snr_20": (1, "20 dB"),
    "snr_10": (2, "10 dB"),
    "snr_5": (3, "5 dB"),
}

FIELDNAMES = [
    "system_id",
    "system_label",
    "system_group",
    "condition",
    "condition_label",
    "condition_order",
    "wer",
    "cer",
    "substitutions",
    "deletions",
    "insertions",
    "source_path",
]


@dataclass(frozen=True)
class ResultSpec:
    system_id: str
    system_label: str
    system_group: str
    metrics_path: Path


@dataclass(frozen=True)
class ResultExportOutputs:
    long_metrics_path: Path
    summary_table_path: Path
    figure_paths: tuple[Path, ...]


def condition_sort_key(condition: str) -> tuple[int, str]:
    return CONDITION_ORDER.get(condition, (99, condition))


def condition_label(condition: str) -> str:
    return CONDITION_ORDER.get(condition, (99, condition))[1]


def collect_metric_rows(specs: list[ResultSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        with Path(spec.metrics_path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                condition = source_row["test_condition"]
                order, label = condition_sort_key(condition)
                rows.append(
                    {
                        "system_id": spec.system_id,
                        "system_label": spec.system_label,
                        "system_group": spec.system_group,
                        "condition": condition,
                        "condition_label": label,
                        "condition_order": order,
                        "wer": float(source_row["wer"]),
                        "cer": float(source_row["cer"]),
                        "substitutions": int(source_row["substitutions"]),
                        "deletions": int(source_row["deletions"]),
                        "insertions": int(source_row["insertions"]),
                        "source_path": str(spec.metrics_path),
                    }
                )
    return sorted(rows, key=lambda row: (row["system_id"], row["condition_order"]))


def export_result_artifacts(
    specs: list[ResultSpec],
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> ResultExportOutputs:
    rows = collect_metric_rows(specs)
    tables_root = Path(tables_dir)
    figures_root = Path(figures_dir)
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    long_metrics_path = tables_root / "metrics_long.csv"
    summary_table_path = tables_root / "metrics_summary.csv"
    _write_long_metrics(long_metrics_path, rows)
    _write_summary_table(summary_table_path, rows)

    figure_paths = (
        figures_root / "wer_vs_snr.png",
        figures_root / "cer_vs_snr.png",
        figures_root / "relative_wer_increase.png",
        figures_root / "edit_breakdown_5db.png",
    )
    plot_metric_curve(rows, metric="wer", output_path=figure_paths[0])
    plot_metric_curve(rows, metric="cer", output_path=figure_paths[1])
    plot_relative_wer_increase(rows, output_path=figure_paths[2])
    plot_edit_breakdown(rows, output_path=figure_paths[3], condition="snr_5")

    return ResultExportOutputs(
        long_metrics_path=long_metrics_path,
        summary_table_path=summary_table_path,
        figure_paths=figure_paths,
    )


def _write_long_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    systems = _system_order(rows)
    by_system_condition = {
        (row["system_id"], row["condition"]): row
        for row in rows
    }
    fieldnames = [
        "system_id",
        "system_label",
        "clean_wer",
        "snr20_wer",
        "snr10_wer",
        "snr5_wer",
        "clean_cer",
        "snr20_cer",
        "snr10_cer",
        "snr5_cer",
        "relative_wer_increase_5db",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system_id in systems:
            clean = by_system_condition[(system_id, "clean")]
            snr5 = by_system_condition[(system_id, "snr_5")]
            clean_wer = clean["wer"]
            relative = ((snr5["wer"] - clean_wer) / clean_wer) if clean_wer > 0 else np.nan
            writer.writerow(
                {
                    "system_id": system_id,
                    "system_label": clean["system_label"],
                    "clean_wer": f"{clean_wer:.6f}",
                    "snr20_wer": f"{by_system_condition[(system_id, 'snr_20')]['wer']:.6f}",
                    "snr10_wer": f"{by_system_condition[(system_id, 'snr_10')]['wer']:.6f}",
                    "snr5_wer": f"{snr5['wer']:.6f}",
                    "clean_cer": f"{clean['cer']:.6f}",
                    "snr20_cer": f"{by_system_condition[(system_id, 'snr_20')]['cer']:.6f}",
                    "snr10_cer": f"{by_system_condition[(system_id, 'snr_10')]['cer']:.6f}",
                    "snr5_cer": f"{snr5['cer']:.6f}",
                    "relative_wer_increase_5db": f"{relative:.6f}",
                }
            )


def _system_order(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        if row["system_id"] not in ordered:
            ordered.append(row["system_id"])
    return ordered


def _system_palette(rows: list[dict[str, Any]]) -> dict[str, str]:
    families = ["orange", "gold", "blue", "olive", "pink"]
    return {
        system_id: COLOR_FAMILIES[families[index % len(families)]]["base"]
        for index, system_id in enumerate(_system_order(rows))
    }


def _use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        },
    )


def _add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=76, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=110, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.78)
    left = ax.get_position().x0
    fig.text(
        left,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(left, 0.91, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def _rows_to_frame(rows: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows).sort_values(["condition_order", "system_id"])


def plot_metric_curve(rows: list[dict[str, Any]], metric: str, output_path: str | Path) -> None:
    _use_chart_theme()
    df = _rows_to_frame(rows)
    palette = _system_palette(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.lineplot(
        data=df,
        x="condition_label",
        y=metric,
        hue="system_id",
        style="system_id",
        markers=True,
        dashes=True,
        palette=palette,
        linewidth=1.4,
        markersize=6,
        ax=ax,
    )
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    handles, labels = ax.get_legend_handles_labels()
    label_map = {row["system_id"]: row["system_label"] for row in rows}
    ax.legend(
        handles,
        [label_map.get(label, label) for label in labels],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
    )
    _add_chart_header(
        fig,
        ax,
        title=f"{metric.upper()} rises as SNR decreases",
        subtitle="Mini LibriSpeech train128/dev32; clean training, evaluated on clean and additive-noise conditions.",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_relative_wer_increase(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    _use_chart_theme()
    import pandas as pd

    df = _rows_to_frame(rows)
    clean = df[df["condition"] == "clean"].set_index("system_id")
    snr5 = df[df["condition"] == "snr_5"].set_index("system_id")
    records = []
    for system_id in _system_order(rows):
        clean_wer = float(clean.loc[system_id, "wer"])
        snr5_wer = float(snr5.loc[system_id, "wer"])
        relative = ((snr5_wer - clean_wer) / clean_wer) if clean_wer > 0 else np.nan
        records.append(
            {
                "system_id": system_id,
                "system_label": clean.loc[system_id, "system_label"],
                "relative_increase": relative,
            }
        )
    plot_df = pd.DataFrame(records).sort_values("relative_increase", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    family = COLOR_FAMILIES["orange"]
    bars = ax.barh(
        plot_df["system_label"],
        plot_df["relative_increase"],
        color=family["base"],
        edgecolor=family["dark"],
        linewidth=1.0,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("WER increase from clean to 5 dB")
    ax.set_ylabel("")
    for bar, value in zip(bars, plot_df["relative_increase"]):
        ax.text(
            value + max(plot_df["relative_increase"].max(), 0.01) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            ha="left",
            fontsize=8,
            color=TOKENS["ink"],
        )
    _add_chart_header(
        fig,
        ax,
        title="Severe noise creates large relative WER increases",
        subtitle="Relative change uses each system's clean WER as denominator; small clean WER makes degradation look large.",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_edit_breakdown(rows: list[dict[str, Any]], output_path: str | Path, condition: str) -> None:
    _use_chart_theme()
    import pandas as pd

    df = _rows_to_frame(rows)
    plot_df = df[df["condition"] == condition].copy()
    plot_df = plot_df.sort_values("wer", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    segments = [
        ("substitutions", COLOR_FAMILIES["blue"]),
        ("deletions", COLOR_FAMILIES["gold"]),
        ("insertions", COLOR_FAMILIES["pink"]),
    ]
    for column, family in segments:
        values = plot_df[column].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            label=column.title(),
            color=family["base"],
            edgecolor=family["dark"],
            linewidth=1.0,
        )
        left += values
    ax.set_yticks(y, plot_df["system_label"])
    ax.set_xlabel("Word edit count")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, borderaxespad=0)
    _add_chart_header(
        fig,
        ax,
        title=f"Word-error composition at {condition_label(condition)}",
        subtitle="Stacked substitutions, deletions, and insertions on the dev32 additive-noise evaluation set.",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
