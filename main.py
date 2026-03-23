"""
run_all.py — Master script to run all experiments, generate plots, and create PDF.

This is the single entry point for the entire project.
Run with: python run_all.py

Steps:
  1. Run baseline experiments (FedAvg IID, FedAvg Non-IID, FedProx Non-IID)
  2. Run μ sensitivity sweep
  3. Run improved experiments (Adaptive, Decaying, Hybrid FedProx)
  4. Generate all comparison plots
  5. Generate final research PDF
"""

import os
import sys
import copy
import yaml
import torch

from src.server import federated_training
from src.utils import (
    set_seed,
    save_results,
    plot_comparison,
    plot_mu_sweep,
    plot_final_comparison,
)


# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(name):
    """Load a YAML config from the configs directory."""
    path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config, device=DEVICE):
    """Run a single experiment and save results."""
    set_seed(config.get("seed", 42))
    history = federated_training(config, device=device)
    results_path = os.path.join(RESULTS_DIR, f"{config['experiment_name']}_results.json")
    save_results(history, results_path)
    return history


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FedProx: Baseline vs Improved — Full Experiment Run   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nDevice: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}\n")

    all_histories = {}

    # ── Step 1: Baseline Experiments ─────────────────────────────────────

    print("\n" + "━" * 60)
    print("  PHASE 1: BASELINE EXPERIMENTS")
    print("━" * 60)

    # 1a. FedAvg IID
    config = load_config("fedavg_iid")
    all_histories["FedAvg (IID)"] = run_experiment(config)

    # 1b. FedAvg Non-IID
    config = load_config("fedavg_noniid")
    all_histories["FedAvg (Non-IID)"] = run_experiment(config)

    # 1c. FedProx Non-IID
    config = load_config("fedprox_noniid")
    all_histories["FedProx (Non-IID)"] = run_experiment(config)

    # ── Step 2: μ Sensitivity Sweep ──────────────────────────────────────

    print("\n" + "━" * 60)
    print("  PHASE 2: μ SENSITIVITY ANALYSIS")
    print("━" * 60)

    sweep_config = load_config("fedprox_mu_sweep")
    mu_values = sweep_config.get("mu_values", [0, 0.001, 0.01, 0.1])
    mu_accuracies = []

    for mu_val in mu_values:
        cfg = copy.deepcopy(sweep_config)
        cfg["mu"] = mu_val
        cfg["experiment_name"] = f"fedprox_mu_{mu_val}"
        # Use fedavg strategy when mu=0
        if mu_val == 0:
            cfg["strategy"] = "fedavg"
        history = run_experiment(cfg)
        mu_accuracies.append(history["accuracy"][-1])

    # ── Step 3: Improved Experiments ─────────────────────────────────────

    print("\n" + "━" * 60)
    print("  PHASE 3: IMPROVED METHODS")
    print("━" * 60)

    # 3a. Adaptive FedProx
    config = load_config("adaptive_fedprox")
    all_histories["Adaptive FedProx"] = run_experiment(config)

    # 3b. Decaying FedProx
    config = load_config("decaying_fedprox")
    all_histories["Decaying FedProx"] = run_experiment(config)

    # 3c. Hybrid FedProx
    config = load_config("hybrid_fedprox")
    all_histories["Hybrid FedProx"] = run_experiment(config)

    # ── Step 4: Generate Plots ───────────────────────────────────────────

    print("\n" + "━" * 60)
    print("  PHASE 4: GENERATING PLOTS")
    print("━" * 60 + "\n")

    # Plot 1: FedAvg IID vs Non-IID (Accuracy)
    plot_comparison(
        {"FedAvg (IID)": all_histories["FedAvg (IID)"],
         "FedAvg (Non-IID)": all_histories["FedAvg (Non-IID)"]},
        metric="accuracy",
        save_path=os.path.join(RESULTS_DIR, "accuracy_fedavg_comparison.png"),
        title="FedAvg: IID vs Non-IID — Accuracy",
    )

    # Plot 2: FedAvg IID vs Non-IID (Loss)
    plot_comparison(
        {"FedAvg (IID)": all_histories["FedAvg (IID)"],
         "FedAvg (Non-IID)": all_histories["FedAvg (Non-IID)"]},
        metric="loss",
        save_path=os.path.join(RESULTS_DIR, "loss_fedavg_comparison.png"),
        title="FedAvg: IID vs Non-IID — Loss",
    )

    # Plot 3: FedAvg vs FedProx Non-IID (Accuracy)
    plot_comparison(
        {"FedAvg (Non-IID)": all_histories["FedAvg (Non-IID)"],
         "FedProx (Non-IID)": all_histories["FedProx (Non-IID)"]},
        metric="accuracy",
        save_path=os.path.join(RESULTS_DIR, "accuracy_fedavg_vs_fedprox.png"),
        title="FedAvg vs FedProx (Non-IID) — Accuracy",
    )

    # Plot 4: μ Sensitivity
    plot_mu_sweep(
        mu_values, mu_accuracies,
        save_path=os.path.join(RESULTS_DIR, "mu_sensitivity.png"),
    )

    # Plot 5: All Methods Accuracy
    plot_comparison(
        all_histories,
        metric="accuracy",
        save_path=os.path.join(RESULTS_DIR, "accuracy_all_methods.png"),
        title="All Methods — Accuracy vs Rounds (Non-IID)",
    )

    # Plot 6: All Methods Loss
    plot_comparison(
        all_histories,
        metric="loss",
        save_path=os.path.join(RESULTS_DIR, "loss_all_methods.png"),
        title="All Methods — Loss vs Rounds (Non-IID)",
    )

    # Plot 7: Final Accuracy Comparison (Bar Chart)
    method_names = list(all_histories.keys())
    final_accuracies = [h["accuracy"][-1] for h in all_histories.values()]
    plot_final_comparison(
        method_names, final_accuracies,
        save_path=os.path.join(RESULTS_DIR, "final_accuracy_comparison.png"),
    )

    # ── Step 5: Generate PDF ─────────────────────────────────────────────

    # print("\n" + "━" * 60)
    # print("  PHASE 5: GENERATING PDF REPORT")
    # print("━" * 60 + "\n")

    # from generate_pdf import generate_report
    # generate_report(all_histories, mu_values, mu_accuracies, RESULTS_DIR)

    # ── Summary ──────────────────────────────────────────────────────────

    # print("\n" + "╔══════════════════════════════════════════════════════════╗")
    # print("║                    EXPERIMENT SUMMARY                    ║")
    # print("╚══════════════════════════════════════════════════════════╝")
    # print(f"\n{'Method':<25} {'Final Accuracy':>15}")
    # print("─" * 42)
    # for name, history in all_histories.items():
    #     acc = history["accuracy"][-1]
    #     print(f"  {name:<23} {acc:>13.4f}")

    # print(f"\n{'μ Value':<25} {'Final Accuracy':>15}")
    # print("─" * 42)
    # for mu, acc in zip(mu_values, mu_accuracies):
    #     print(f"  μ = {mu:<19} {acc:>13.4f}")

    # print(f"\n✅ All plots saved to: {RESULTS_DIR}/")
    # print(f"✅ PDF report: {os.path.join(RESULTS_DIR, 'FedProx_Research_Paper.pdf')}")
    # print(f"\n{'='*60}")
    # print("  ALL DONE! 🎉")
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
