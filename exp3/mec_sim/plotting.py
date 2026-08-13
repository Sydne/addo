from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from .models import SimulationSummary

plt.rcParams['font.family'] = 'Times New Roman'


def plot_bar_comparison(
    summaries: List[SimulationSummary],
    metric: str,
    title: str,
    ylabel: str,
    save_path: str,
) -> None:
    names = [summary.name for summary in summaries]
    values = [getattr(summary, metric) for summary in summaries]
    plt.figure(figsize=(8, 4.8))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(rotation=20, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_line(
    xs: List[float],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    save_path: str,
    marker: str = "o",
    series_label: str | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    reference_style: str = "--",
    reference_color: str = "gray",
) -> None:
    plt.figure(figsize=(7, 4.8))
    plt.plot(xs, ys, marker=marker, label=series_label)
    if reference_y is not None:
        plt.axhline(
            reference_y,
            linestyle=reference_style,
            color=reference_color,
            linewidth=1.2,
            label=reference_label,
        )
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    if series_label is not None or reference_label is not None:
        plt.legend(loc="best", fontsize=14, framealpha=0.9, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_two_lines(
    xs: List[float],
    ys1: List[float],
    ys2: List[float],
    xlabel: str,
    ylabel1: str,
    ylabel2: str,
    save_path: str,
    marker1: str = "o",
    marker2: str = "s",
    color1: str = "tab:blue",
    color2: str = "tab:red",
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    
    ax1.plot(xs, ys1, marker=marker1, color=color1, label=ylabel1)
    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.set_ylabel(ylabel1, fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    
    ax2 = ax1.twinx()
    ax2.plot(xs, ys2, marker=marker2, color=color2, label=ylabel2)
    ax2.set_ylabel(ylabel2, fontsize=20)
    ax2.tick_params(axis='y', labelsize=16)
        
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        fontsize=12,
        framealpha=0.9,
        facecolor="white",
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_line(
    series: Dict[str, Tuple[List[float], List[float]]],
    xlabel: str,
    ylabel: str,
    save_path: str,
    figure_size: Tuple[float, float] = (7, 4.8),
    legend_columns: int = 1,
    legend_location: str = "best",
    legend_bbox_to_anchor: Tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=figure_size)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (label, (xs, ys)) in enumerate(series.items()):
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        series_len = len(xs)
        if series_len <= 12:
            markevery = 1
        else:
            step = max(series_len // 12, 1)
            markevery = (idx % step, step)

        plt.plot(
            xs,
            ys,
            marker=marker,
            linestyle=linestyle,
            label=label,
            linewidth=2,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markevery=markevery,
        )
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis="both", labelsize=16)
    legend_kwargs = {
        "loc": legend_location,
        "fontsize": 16,
        "framealpha": 0.9,
        "facecolor": "white",
        "ncol": legend_columns,
    }
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    plt.legend(**legend_kwargs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
