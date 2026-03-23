"""
client.py — Local training logic for federated learning clients.

Supports five training strategies:
  - fedavg:    Standard local SGD, no proximal term
  - fedprox:   Fixed μ proximal regularization
  - adaptive:  Drift-aware μ (μ = α · ||w - w_global||)
  - decaying:  Time-decaying μ (μ = μ₀ · e^(-β·t))
  - hybrid:    Combined adaptive + decaying (μ = α · ||w - w_global|| · e^(-β·t))
"""

import copy
import math
import torch
import torch.nn as nn


def compute_proximal_term(model, global_params):
    """
    Compute the proximal penalty: sum of ||w_local - w_global||² for all parameters.

    This penalizes the local model for drifting too far from the global model,
    which is the core mechanism of FedProx.

    Args:
        model: Local model being trained
        global_params: Dict of global model parameters (state_dict)

    Returns:
        Scalar tensor representing the proximal penalty
    """
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        global_param = global_params[name].to(param.device)
        proximal_term += torch.sum((param - global_param) ** 2)
    return proximal_term


def compute_drift(model, global_params):
    """
    Compute the L2 norm of model drift: ||w_local - w_global||.

    Used by adaptive and hybrid strategies to scale μ proportionally to drift.

    Args:
        model: Local model after training
        global_params: Dict of global model parameters

    Returns:
        Float value of the drift norm
    """
    drift = 0.0
    for name, param in model.named_parameters():
        global_param = global_params[name].to(param.device)
        drift += torch.sum((param.data - global_param) ** 2).item()
    return math.sqrt(drift)


def get_mu(strategy, config, model=None, global_params=None, current_round=0):
    """
    Compute the proximal coefficient μ based on the chosen strategy.

    Args:
        strategy: One of "fedavg", "fedprox", "adaptive", "decaying", "hybrid"
        config: Dictionary with hyperparameters (mu, alpha_drift, beta_decay, mu_initial)
        model: Current local model (needed for adaptive/hybrid)
        global_params: Global model state dict (needed for adaptive/hybrid)
        current_round: Current communication round (needed for decaying/hybrid)

    Returns:
        Float value of μ
    """
    if strategy == "fedavg":
        return 0.0

    elif strategy == "fedprox":
        return config.get("mu", 0.01)

    elif strategy == "adaptive":
        # μ_i = α · ||w_i - w_global||
        alpha = config.get("alpha_drift", 0.1)
        if model is not None and global_params is not None:
            drift = compute_drift(model, global_params)
            return alpha * drift
        return config.get("mu", 0.01)

    elif strategy == "decaying":
        # μ_t = μ₀ · e^(-β·t)
        mu_0 = config.get("mu_initial", 0.01)
        beta = config.get("beta_decay", 0.1)
        return mu_0 * math.exp(-beta * current_round)

    elif strategy == "hybrid":
        # μ_{i,t} = α · ||w_i - w_global|| · e^(-β·t)
        alpha = config.get("alpha_drift", 0.1)
        mu_0 = config.get("mu_initial", 0.01)
        beta = config.get("beta_decay", 0.1)
        decay_factor = math.exp(-beta * current_round)
        if model is not None and global_params is not None:
            drift = compute_drift(model, global_params)
            return alpha * drift * decay_factor
        return mu_0 * decay_factor

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def client_train(model, train_loader, global_params, strategy, config,
                 current_round=0, device="cpu"):
    """
    Perform local training on a single client.

    Implements the FedProx objective:
        L = L_task(w) + (μ/2) · ||w - w_global||²

    where μ depends on the chosen strategy.

    Args:
        model: Local model to train (will be modified in-place)
        train_loader: DataLoader for this client's local data
        global_params: State dict of the global model (for proximal term)
        strategy: Training strategy name
        config: Hyperparameter dictionary
        current_round: Current FL round number
        device: Torch device

    Returns:
        dict with:
            - "state_dict": Updated model parameters
            - "train_loss": Average training loss
            - "num_samples": Number of training samples
            - "mu_used": The μ value used during training
    """
    model.train()
    model.to(device)

    # Store global params on device for proximal computation
    global_params_device = {
        k: v.clone().detach().to(device) for k, v in global_params.items()
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    criterion = nn.CrossEntropyLoss()
    local_epochs = config.get("local_epochs", 3)

    total_loss = 0.0
    total_samples = 0
    mu_used = 0.0

    for epoch in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass — task loss
            outputs = model(batch_x)
            task_loss = criterion(outputs, batch_y)

            # Compute μ for this step (adaptive strategies use current model state)
            mu = get_mu(strategy, config, model, global_params_device, current_round)
            mu_used = mu  # Track the last μ used

            # Proximal term: (μ/2) · ||w - w_global||²
            if mu > 0:
                prox_term = compute_proximal_term(model, global_params_device)
                loss = task_loss + (mu / 2.0) * prox_term
            else:
                loss = task_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += task_loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return {
        "state_dict": copy.deepcopy(model.state_dict()),
        "train_loss": avg_loss,
        "num_samples": total_samples,
        "mu_used": mu_used,
    }
