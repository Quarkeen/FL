"""
utils.py — Utility functions for plotting, metrics, and reproducibility.

Provides:
  - Seed setting for reproducibility
  - Comparison plots (multi-curve accuracy/loss vs rounds)
  - μ sensitivity bar charts
  - Final accuracy comparison bar charts
  - Results serialization (JSON)
"""

import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ── Plot styling ─────────────────────────────────────────────────────────────

# Professional color palette
COLORS = {
    "FedAvg (IID)": "#2196F3",
    "FedAvg (Non-IID)": "#FF9800",
    "FedProx (Non-IID)": "#4CAF50",
    "Adaptive FedProx": "#E91E63",
    "Decaying FedProx": "#9C27B0",
    "Hybrid FedProx": "#00BCD4",
}

STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "adaptive": "Adaptive FedProx",
    "decaying": "Decaying FedProx",
    "hybrid": "Hybrid FedProx",
}


def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _style_plot(ax, title, xlabel, ylabel):
    """Apply consistent styling to a plot."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_comparison(histories, metric, save_path, title=None, labels=None):
    """
    Plot multiple experiment curves on the same axes.

    Args:
        histories: Dict of {label: history_dict} where history_dict has
                   "rounds" and the given metric key
        metric: Key to plot from history ("accuracy" or "loss")
        save_path: File path to save the PNG
        title: Optional plot title override
        labels: Optional label overrides
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, history) in enumerate(histories.items()):
        display_label = labels.get(label, label) if labels else label
        color = list(COLORS.values())[i % len(COLORS)]
        ax.plot(
            history["rounds"],
            history[metric],
            marker="o",
            markersize=4,
            linewidth=2,
            label=display_label,
            color=color,
        )

    metric_name = metric.capitalize()
    default_title = f"{metric_name} vs Communication Rounds"
    _style_plot(ax, title or default_title, "Communication Round", metric_name)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_mu_sweep(mu_values, accuracies, save_path):
    """
    Bar chart showing final accuracy for different μ values.

    Args:
        mu_values: List of μ values tested
        accuracies: Corresponding final accuracies
        save_path: File path to save the PNG
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = [str(mu) for mu in mu_values]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(mu_values)))

    bars = ax.bar(x_labels, accuracies, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    _style_plot(ax, "μ Sensitivity Analysis", "Proximal Coefficient (μ)", "Final Accuracy")
    ax.legend().remove()  # No legend needed for bar chart

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_final_comparison(method_names, accuracies, save_path):
    """
    Horizontal bar chart comparing final accuracy of all methods.

    Args:
        method_names: List of method names
        accuracies: Corresponding final accuracies
        save_path: File path to save the PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))
    y_pos = range(len(method_names))

    bars = ax.barh(y_pos, accuracies, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}",
            ha="left", va="center", fontsize=10, fontweight="bold"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=11)
    ax.set_xlabel("Final Accuracy", fontsize=12)
    ax.set_title("Final Accuracy Comparison: All Methods", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    # Set x-axis range to zoom in on differences
    min_acc = min(accuracies)
    ax.set_xlim(max(0, min_acc - 0.02), 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


# ── Serialization ────────────────────────────────────────────────────────────

def save_results(history, path):
    """
    Save experiment history to a JSON file.

    Converts non-serializable types and saves config + metrics.
    """
    # Create serializable copy
    serializable = {}
    for key, value in history.items():
        if key == "config":
            serializable[key] = value
        elif isinstance(value, list):
            serializable[key] = [
                v if not isinstance(v, (np.floating, np.integer)) else float(v)
                for v in value
            ]
        else:
            serializable[key] = value

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  💾 Saved results: {path}")


def load_results(path):
    """Load experiment history from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
