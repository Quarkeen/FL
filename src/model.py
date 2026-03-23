"""
model.py — CNN model for MNIST classification in Federated Learning.

Architecture matches the original FedProx paper setup:
  Input(28×28) → Conv(32) → Conv(64) → MaxPool → FC(128) → Output(10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    Used as the shared model architecture across all federated clients.
    """

    def __init__(self):
        super(MNISTNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28×28 → 28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28×28 → 28×28
        self.pool = nn.MaxPool2d(2, 2)                             # 28×28 → 14×14
        self.dropout1 = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def create_model(device="cpu"):
    """Create and return a new MNISTNet model on the specified device."""
    return MNISTNet().to(device)
