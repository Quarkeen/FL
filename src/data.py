"""
data.py — Dataset loading and partitioning for federated learning.

Supports three partitioning strategies:
  - IID: Random equal split across clients
  - Non-IID: Each client gets a fixed subset of digit classes
  - Dirichlet: Heterogeneity controlled by alpha parameter
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir="./data"):
    """
    Download and load MNIST dataset with standard normalization.
    Returns (train_dataset, test_dataset).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def iid_partition(dataset, num_clients):
    """
    IID partitioning: randomly split dataset into equal-sized subsets.
    Each client gets a representative sample of all classes.

    Args:
        dataset: Full training dataset
        num_clients: Number of federated clients

    Returns:
        List of DataLoader objects, one per client
    """
    total = len(dataset)
    indices = np.random.permutation(total)
    split_size = total // num_clients

    client_loaders = []
    for i in range(num_clients):
        start = i * split_size
        end = start + split_size if i < num_clients - 1 else total
        client_indices = indices[start:end].tolist()
        subset = Subset(dataset, client_indices)
        client_loaders.append(subset)

    return client_loaders


def noniid_partition(dataset, num_clients, classes_per_client=2):
    """
    Non-IID partitioning: each client gets data from only a fixed number of classes.

    For 10 clients with 2 classes each:
      Client 0: classes {0,1}, Client 1: classes {2,3}, ... Client 4: classes {8,9}
    For >5 clients, classes are assigned round-robin.

    Args:
        dataset: Full training dataset
        num_clients: Number of federated clients
        classes_per_client: Number of digit classes per client

    Returns:
        List of Subset objects, one per client
    """
    targets = np.array(dataset.targets)
    num_classes = 10

    # Group indices by class
    class_indices = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}

    # Assign classes to clients in round-robin fashion
    client_subsets = []
    for i in range(num_clients):
        assigned_classes = []
        for j in range(classes_per_client):
            class_id = (i * classes_per_client + j) % num_classes
            assigned_classes.append(class_id)

        # Gather indices for assigned classes
        client_indices = []
        for c in assigned_classes:
            client_indices.extend(class_indices[c])

        np.random.shuffle(client_indices)
        client_subsets.append(Subset(dataset, client_indices))

    return client_subsets


def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """
    Dirichlet-based Non-IID partitioning: heterogeneity controlled by alpha.

    alpha → 0: extreme non-IID (each client gets ~1 class)
    alpha → ∞: approaches IID
    alpha = 0.5: moderate heterogeneity (recommended default)

    Args:
        dataset: Full training dataset
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter

    Returns:
        List of Subset objects, one per client
    """
    targets = np.array(dataset.targets)
    num_classes = 10

    # Group indices by class
    class_indices = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}

    # Generate Dirichlet proportions for each class
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Ensure minimum allocation (at least 1 sample per client if possible)
        proportions = proportions / proportions.sum()
        splits = (proportions * len(indices)).astype(int)

        # Distribute remaining samples due to rounding
        remainder = len(indices) - splits.sum()
        for j in range(int(remainder)):
            splits[j % num_clients] += 1

        # Assign indices to clients
        start = 0
        for i in range(num_clients):
            end = start + splits[i]
            client_indices[i].extend(indices[start:end])
            start = end

    # Create Subset objects
    client_subsets = []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_subsets.append(Subset(dataset, client_indices[i]))

    return client_subsets


def create_client_loaders(dataset, num_clients, partition_type="iid",
                          batch_size=32, classes_per_client=2, dirichlet_alpha=0.5):
    """
    Convenience function to create client DataLoaders based on partition type.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        partition_type: "iid", "noniid", or "dirichlet"
        batch_size: Batch size for DataLoaders
        classes_per_client: Classes per client (for noniid)
        dirichlet_alpha: Alpha for Dirichlet partition

    Returns:
        List of DataLoader objects, one per client
    """
    if partition_type == "iid":
        subsets = iid_partition(dataset, num_clients)
    elif partition_type == "noniid":
        subsets = noniid_partition(dataset, num_clients, classes_per_client)
    elif partition_type == "dirichlet":
        subsets = dirichlet_partition(dataset, num_clients, dirichlet_alpha)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    loaders = []
    for subset in subsets:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders
