"""
server.py — Federated learning server: orchestrates training rounds.

Handles:
  - Global model initialization
  - Distribution of weights to clients
  - Collection and weighted aggregation of client updates
  - Evaluation on the global test set
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import create_model
from src.data import load_mnist, create_client_loaders
from src.client import client_train


def aggregate(client_results):
    """
    Weighted FedAvg aggregation of client model updates.

    Weights each client's update by its number of training samples,
    producing a weighted average of all model parameters.

    Args:
        client_results: List of dicts from client_train(), each containing
                       "state_dict" and "num_samples"

    Returns:
        Aggregated state_dict (OrderedDict)
    """
    total_samples = sum(r["num_samples"] for r in client_results)

    # Initialize aggregated state dict with zeros
    agg_state = copy.deepcopy(client_results[0]["state_dict"])
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)

    # Weighted sum
    for result in client_results:
        weight = result["num_samples"] / total_samples
        for key in agg_state:
            agg_state[key] += result["state_dict"][key].float() * weight

    return agg_state


def evaluate(model, test_loader, device="cpu"):
    """
    Evaluate model accuracy and loss on a test dataset.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Torch device

    Returns:
        dict with "accuracy" (float 0-1) and "loss" (float)
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return {"accuracy": accuracy, "loss": avg_loss}


def federated_training(config, device="cpu", verbose=True):
    """
    Main federated learning loop.

    Implements the standard FL protocol:
      1. Server initializes global model
      2. For each round:
         a. Distribute global weights to all clients
         b. Each client trains locally
         c. Server aggregates updates (weighted average)
         d. Evaluate on global test set
      3. Return training history

    Args:
        config: Dictionary with experiment configuration:
            - experiment_name, strategy, num_clients, num_rounds,
            - local_epochs, batch_size, lr, mu, partition, seed, etc.
        device: Torch device ("cpu" or "cuda")
        verbose: Whether to print progress

    Returns:
        dict with:
            - "rounds": list of round numbers
            - "accuracy": list of accuracies per round
            - "loss": list of losses per round
            - "client_losses": list of per-client loss lists per round
            - "client_mus": list of per-client μ values per round
            - "config": the config dict used
    """
    experiment_name = config.get("experiment_name", "unnamed")
    strategy = config.get("strategy", "fedavg")
    num_clients = config.get("num_clients", 10)
    num_rounds = config.get("num_rounds", 20)
    batch_size = config.get("batch_size", 32)
    partition = config.get("partition", "iid")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Strategy: {strategy} | Clients: {num_clients} | Rounds: {num_rounds}")
        print(f"  Partition: {partition}")
        print(f"{'='*60}")

    # Load dataset
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create client data loaders
    client_loaders = create_client_loaders(
        train_dataset,
        num_clients,
        partition_type=partition,
        batch_size=batch_size,
        classes_per_client=config.get("classes_per_client", 2),
        dirichlet_alpha=config.get("dirichlet_alpha", 0.5),
    )

    # Initialize global model
    global_model = create_model(device)
    global_params = copy.deepcopy(global_model.state_dict())

    # Training history
    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "client_losses": [],
        "client_mus": [],
        "config": config,
    }

    # Federated training loop
    for round_num in range(1, num_rounds + 1):
        client_results = []
        round_mus = []

        for client_id in range(num_clients):
            # Create a fresh local model with global weights
            local_model = create_model(device)
            local_model.load_state_dict(copy.deepcopy(global_params))

            # Local training
            result = client_train(
                model=local_model,
                train_loader=client_loaders[client_id],
                global_params=global_params,
                strategy=strategy,
                config=config,
                current_round=round_num,
                device=device,
            )

            client_results.append(result)
            round_mus.append(result["mu_used"])

        # Aggregate client updates → new global model
        global_params = aggregate(client_results)
        global_model.load_state_dict(global_params)

        # Evaluate global model
        eval_result = evaluate(global_model, test_loader, device)

        # Record history
        history["rounds"].append(round_num)
        history["accuracy"].append(eval_result["accuracy"])
        history["loss"].append(eval_result["loss"])
        history["client_losses"].append([r["train_loss"] for r in client_results])
        history["client_mus"].append(round_mus)

        if verbose:
            avg_mu = sum(round_mus) / len(round_mus)
            avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
            print(f"  Round {round_num:3d}/{num_rounds} | "
                  f"Acc: {eval_result['accuracy']:.4f} | "
                  f"Loss: {eval_result['loss']:.4f} | "
                  f"Avg Client Loss: {avg_client_loss:.4f} | "
                  f"Avg μ: {avg_mu:.6f}")

    if verbose:
        print(f"\n  Final Accuracy: {history['accuracy'][-1]:.4f}")
        print(f"{'='*60}\n")

    return history
