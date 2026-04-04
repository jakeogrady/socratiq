from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

BLUE = "#2166AC"
ORANGE = "#D6604D"
GREEN = "#4DAC26"
GREY = "#808080"

OUTPUT_DIR = "outputs/"


def save_figure(figure: Figure, filename: str) -> None:
    """Save a figure as both PDF and PNG and close it."""
    figure.tight_layout()
    figure.savefig(f"{OUTPUT_DIR}{filename}.pdf")
    figure.savefig(f"{OUTPUT_DIR}{filename}.png")
    plt.close(figure)
    logger.info("%s saved.", filename)


def plot_baseline_comparison() -> None:
    """Plot baseline GSM8K accuracy for all evaluated models as a horizontal bar chart."""
    model_names: list[str] = [
        "LLaMA-3.2-1B",
        "Qwen3-0.6B (Base)",
        "Mistral-7B-Instruct\n(8-bit quantised)",
        "Qwen3-1.7B (Base)",
        "LLaMA-3.2-3B-Instruct",
        "Gemma-3-4B-it",
        "Qwen3-4B-bf16",
    ]
    accuracy_values: list[float] = [26.08, 36.47, 47.15, 53.53, 56.41, 75.13, 78.62]
    bar_colours: list[str] = [ORANGE, BLUE, GREY, BLUE, ORANGE, ORANGE, BLUE]
    legend_labels: dict[str, str] = {
        BLUE: "Base model",
        ORANGE: "Instruction-tuned",
        GREY: "Instruction-tuned (8-bit quantised)",
    }

    figure, ax = plt.subplots(figsize=(10, 5))
    ax: Axes

    bars = ax.barh(
        model_names,
        accuracy_values,
        color=bar_colours,
        edgecolor="white",
        linewidth=0.6,
        height=0.55,
    )

    for bar, accuracy in zip(bars, accuracy_values, strict=True):
        ax.text(
            accuracy + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.2f}%",
            va="center",
            ha="left",
        )

    ax.set_xlabel("Exact Match Accuracy (%)  —  GSM8K, 4-shot, 1 sample")
    ax.set_title("Baseline Model Performance on GSM8K")
    ax.set_xlim(0, 92)

    legend_patches = [
        mpatches.Patch(color=colour, label=legend_label)
        for colour, legend_label in legend_labels.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    save_figure(figure, "fig1_baseline_comparison")


def plot_self_consistency_scaling() -> None:
    """Plot accuracy vs self-consistency sample count for MultiArith and SVAMP."""
    sample_counts: list[int] = [1, 2, 4]

    multiarith_base_06b: list[float] = [45.00, 41.00, 60.00]
    multiarith_finetuned_06b: list[float] = [79.44, 79.44, 92.22]
    multiarith_base_17b: list[float] = [48.33, 41.67, 54.44]
    multiarith_finetuned_17b: list[float] = [94.44, 97.22, 98.89]

    svamp_base_06b: list[float] = [29.00, 33.00, 42.33]
    svamp_finetuned_06b: list[float] = [45.33, 43.67, 52.67]
    svamp_base_17b: list[float] = [39.67, 33.33, 45.33]
    svamp_finetuned_17b: list[float] = [65.33, 67.00, 73.00]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    benchmark_data: list[tuple] = [
        (
            "MultiArith",
            multiarith_base_06b,
            multiarith_finetuned_06b,
            multiarith_base_17b,
            multiarith_finetuned_17b,
        ),
        (
            "SVAMP",
            svamp_base_06b,
            svamp_finetuned_06b,
            svamp_base_17b,
            svamp_finetuned_17b,
        ),
    ]

    ax: Axes
    for ax, (benchmark_title, base_06b, finetuned_06b, base_17b, finetuned_17b) in zip(
        axes, benchmark_data, strict=True
    ):
        ax.plot(
            sample_counts,
            base_06b,
            linestyle="--",
            marker="o",
            lw=1.8,
            ms=7,
            color=BLUE,
            label="Base Qwen3-0.6B",
        )
        ax.plot(
            sample_counts,
            finetuned_06b,
            linestyle="-",
            marker="o",
            lw=2.2,
            ms=7,
            color=BLUE,
            label="Fine-tuned Qwen3-0.6B",
        )
        ax.plot(
            sample_counts,
            base_17b,
            linestyle="--",
            marker="s",
            lw=1.8,
            ms=7,
            color=ORANGE,
            label="Base Qwen3-1.7B",
        )
        ax.plot(
            sample_counts,
            finetuned_17b,
            linestyle="-",
            marker="s",
            lw=2.2,
            ms=7,
            color=ORANGE,
            label="Fine-tuned Qwen3-1.7B",
        )

        ax.set_title(benchmark_title)
        ax.set_xlabel("Number of Self-Consistency Samples")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(sample_counts)
        ax.set_xticklabels(["1", "2", "4"])
        ax.set_ylim(20, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.06),
        framealpha=0.9,
    )
    figure.text(
        0.5,
        1.01,
        "Solid lines = fine-tuned  ·  Dashed lines = base  ·  "
        "Blue = 0.6B  ·  Orange = 1.7B",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )
    figure.suptitle(
        "Self-Consistency Scaling: Accuracy vs Number of Samples",
        fontsize=13,
        fontweight="bold",
        y=1.07,
    )

    save_figure(figure, "fig2_self_consistency_scaling")


def plot_finetuning_gains() -> None:
    """Plot GSM8K exact match accuracy for base and fine-tuned models as grouped bars."""
    model_names: list[str] = ["Qwen3-0.6B", "Qwen3-1.7B"]
    base_accuracies: list[float] = [36.47, 53.53]
    finetuned_accuracies: list[float] = [49.05, 66.49]
    accuracy_gains: list[float] = [
        finetuned - base
        for finetuned, base in zip(finetuned_accuracies, base_accuracies, strict=True)
    ]

    x_positions: np.ndarray = np.arange(len(model_names))
    bar_width: float = 0.32

    figure, ax = plt.subplots(figsize=(7, 5))
    ax: Axes

    base_bars = ax.bar(
        x_positions - bar_width / 2,
        base_accuracies,
        bar_width,
        label="Base model",
        color=BLUE,
        alpha=0.75,
        edgecolor="white",
    )
    finetuned_bars = ax.bar(
        x_positions + bar_width / 2,
        finetuned_accuracies,
        bar_width,
        label="Fine-tuned (Socratic LoRA)",
        color=GREEN,
        alpha=0.9,
        edgecolor="white",
    )

    for bar, accuracy in zip(base_bars, base_accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.6,
            f"{accuracy:.2f}%",
            ha="center",
        )

    for bar, accuracy, gain in zip(
        finetuned_bars, finetuned_accuracies, accuracy_gains, strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.6,
            f"{accuracy:.2f}%",
            ha="center",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 3.8,
            f"+{gain:.2f}pp",
            ha="center",
            fontsize=9,
            color="#2e7d18",
            fontweight="bold",
        )

    ax.set_ylabel("Exact Match Accuracy (%)  —  GSM8K, 4-shot, 1 sample")
    ax.set_title("Fine-tuning Gains on GSM8K: Base vs Socratic LoRA")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend()

    save_figure(figure, "fig3_finetuning_gains_gsm8k")


def plot_lora_runs() -> None:
    """Plot GSM8K EM accuracy across the three Qwen3-0.6B LoRA fine-tuning runs."""
    run_labels: list[str] = [
        "Run 1\n12 layers · rank 16\nLR = 1×10⁻⁴",
        "Run 2\n28 layers · rank 16\nLR = 8×10⁻⁵",
        "Run 3\n28 layers · rank 32\nLR = 5×10⁻⁵",
    ]
    em_accuracies: list[float] = [36.47, 49.05, 49.05]
    bar_colours: list[str] = [ORANGE, GREEN, GREEN]

    figure, ax = plt.subplots(figsize=(8, 5))
    ax: Axes

    bars = ax.bar(
        run_labels,
        em_accuracies,
        color=bar_colours,
        edgecolor="white",
        width=0.45,
    )

    for bar, accuracy in zip(bars, em_accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.5,
            f"{accuracy:.2f}%",
            ha="center",
            fontweight="bold",
        )

    baseline_line = ax.axhline(
        36.47,
        linestyle="--",
        lw=1.8,
        color=BLUE,
        label="Base model baseline (36.47%)",
    )

    ax.set_ylabel("Exact Match Accuracy (%)  —  GSM8K, 4-shot, 1 sample")
    ax.set_title("Qwen3-0.6B LoRA Runs: Effect of Layer Depth and Rank on EM Accuracy")
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    no_improvement_patch = mpatches.Patch(
        color=ORANGE, label="No improvement over baseline"
    )
    improvement_patch = mpatches.Patch(color=GREEN, label="Significant improvement")
    ax.legend(
        handles=[baseline_line, no_improvement_patch, improvement_patch],
        loc="upper left",
    )

    save_figure(figure, "fig4_lora_run_comparison")


def main() -> None:
    """Generate and save all four project figures."""
    plot_baseline_comparison()
    plot_self_consistency_scaling()
    plot_finetuning_gains()
    plot_lora_runs()
    logger.info("All figures saved successfully.")


if __name__ == "__main__":
    main()
