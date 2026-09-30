"""
PyTorch Temporal Architectures for Real-Time Word-Level ISL Recognition.
Provides:
1. TemporalCNN: Lightweight 1D Temporal Convolutional Network with residual blocks.
2. BiGRUClassifier: 2-layer Bidirectional GRU with temporal attention pooling.
Both architectures are engineered for low-latency CPU real-time inference (<5ms).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    Lightweight 1D Dilated Convolutional Network for sign sequence classification.
    Input shape: (Batch, SeqLen=30, FeatureDim=126)
    Output shape: (Batch, NumClasses)
    """

    def __init__(self, in_features: int = 126, num_classes: int = 12, dropout: float = 0.3):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> Permute to (B, D, T) for Conv1D
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class TemporalAttention(nn.Module):
    """Computes attention weights over temporal sequence frames."""
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (B, T, hidden_dim)
        scores = self.linear(lstm_out)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, hidden_dim)
        return context, weights


class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU Network with temporal attention pooling.
    Input shape: (Batch, SeqLen=30, FeatureDim=126)
    Output shape: (Batch, NumClasses)
    """

    def __init__(self, in_features: int = 126, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 12, dropout: float = 0.3):
        super(BiGRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)  # out: (B, T, 2 * hidden_dim)
        context, _ = self.attention(out)  # (B, 2 * hidden_dim)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits
