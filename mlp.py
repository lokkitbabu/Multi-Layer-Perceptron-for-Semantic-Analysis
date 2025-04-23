import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, hamming_loss

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Sigmoid for multi-label
        )

    def forward(self, x):
        return self.net(x)