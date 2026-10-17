"""Publication-quality matplotlib figures from experimental artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.models import MODEL_DISPLAY_NAMES
from src.utils import ensure_dir, resolve_path

MODEL_ORDER = ["simple_rnn", "bidirectional_rnn", "lstm", "gru"]
COLORS = {
    "simple_rnn": "#4C78A8",
    "bidirectional_rnn": "#F58518",
    "lstm": "#54A24B",
    "gru": "#E45756",
}


def _display(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, figures_dir: Path) -> list[str]:
    _style()
    saved = []
    for metric, ylabel, filename, title in [
        ("train_loss", "Cross-entropy loss", "training_loss_curves.png", "Training loss by epoch"),
        ("val_loss", "Cross-entropy loss", "validation_loss_curves.png", "Validation loss by epoch"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in MODEL_ORDER:
            subset = history[history["model"] == model].sort_values("epoch")
            if subset.empty:
                continue
            ax.plot(
                subset["epoch"],
                subset[metric],
                marker="o",
                label=_display(model),
                color=COLORS[model],
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        path = figures_dir / filename
        _save(fig, path)
        saved.append(str(path))
        history.to_csv(figures_dir / f"{filename.replace('.png', '')}_data.csv", index=False)
    return saved


def plot_bar_comparison(
    test_metrics: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    values = []
    colors = []
    for model in MODEL_ORDER:
        row = test_metrics[test_metrics["model"] == model]
        if row.empty:
            continue
        names.append(_display(model))
        values.append(float(row.iloc[0][column]))
        colors.append(COLORS[model])
    ax.bar(names, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    path = figures_dir / filename
    _save(fig, path)
    pd.DataFrame({"model": names, column: values}).to_csv(
        figures_dir / f"{filename.replace('.png', '')}_data.csv", index=False
    )
    return str(path)


def plot_temperature_metric(
    generation_metrics: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        subset = generation_metrics[generation_metrics["model"] == model].sort_values("temperature")
        if subset.empty:
            continue
        ax.plot(
            subset["temperature"],
            subset[column],
            marker="o",
            label=_display(model),
            color=COLORS[model],
        )
    ax.set_title(title)
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel(ylabel)
    ax.legend()
    path = figures_dir / filename
    _save(fig, path)
    generation_metrics.to_csv(figures_dir / f"{filename.replace('.png', '')}_data.csv", index=False)
    return str(path)


def generate_all_figures(
    history_csv: str | Path,
    test_metrics_csv: str | Path,
    generation_metrics_csv: str | Path,
    figures_dir: str | Path,
) -> list[str]:
    figures_dir = ensure_dir(figures_dir)
    history = pd.read_csv(resolve_path(history_csv))
    test_metrics = pd.read_csv(resolve_path(test_metrics_csv))
    generation_metrics = pd.read_csv(resolve_path(generation_metrics_csv))
    saved = plot_training_curves(history, figures_dir)
    saved.append(
        plot_bar_comparison(
            test_metrics, "test_perplexity", "Test perplexity", "Perplexity (lower is better)",
            "test_perplexity_comparison.png", figures_dir,
        )
    )
    saved.append(
        plot_bar_comparison(
            test_metrics, "test_accuracy", "Test next-word accuracy", "Accuracy (higher is better)",
            "test_accuracy_comparison.png", figures_dir,
        )
    )
    saved.append(
        plot_bar_comparison(
            test_metrics, "total_training_time_sec", "Total training time", "Seconds",
            "training_time_comparison.png", figures_dir,
        )
    )
    saved.append(
        plot_bar_comparison(
            test_metrics, "parameter_count", "Trainable parameter count", "Parameters",
            "parameter_count_comparison.png", figures_dir,
        )
    )
    saved.append(
        plot_temperature_metric(
            generation_metrics, "distinct_1", "Temperature vs lexical diversity (Distinct-1)",
            "Distinct-1", "temperature_vs_diversity.png", figures_dir,
        )
    )
    saved.append(
        plot_temperature_metric(
            generation_metrics, "repeated_bigram_ratio", "Temperature vs repetition (repeated bigram ratio)",
            "Repeated bigram ratio", "temperature_vs_repetition.png", figures_dir,
        )
    )
    return saved
