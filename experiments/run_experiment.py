"""
run_experiment.py — Run a single federated learning experiment from a YAML config.

Usage:
    python experiments/run_experiment.py configs/fedavg_iid.yaml

Loads the config, runs the federated training loop, and saves results to JSON.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.server import federated_training
from src.utils import set_seed, save_results


def run_from_config(config_path, device="cpu"):
    """
    Load a YAML config and run the federated training experiment.

    Args:
        config_path: Path to YAML config file
        device: Torch device

    Returns:
        Training history dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Run experiment
    history = federated_training(config, device=device)

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(config_path)),
        "results_corrected"
    )
    results_path = os.path.join(results_dir, f"{config['experiment_name']}_results.json")
    save_results(history, results_path)

    return history


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/run_experiment.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_from_config(config_path, device)
