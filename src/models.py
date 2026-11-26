"""Arquiteturas usadas no projeto. Iguais às definidas nos notebooks 03 (CNN) e 04 (RNN).

Ficam aqui para poderem ser usadas pelo treino com várias seeds, pela avaliação no teste e pela previsão do bónus.
"""
import torch
from torch import nn

from src.metrics import CLASSES


class SmallCNN(nn.Module):
    """CNN 2D sobre log-mel (1, 64, 313): 4 blocos Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2,
    global average pooling, dropout, camada linear com 3 saídas. ~98 mil parâmetros."""

    def __init__(self, channels=(16, 32, 64, 128), dropout=0.3, n_classes=len(CLASSES)):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class RNNClassifier(nn.Module):
    """RNN bidirecional sobre MFCC (313, n_features), pooling média + máximo no tempo, camada linear com 3 saídas.

    cell="gru" com os valores por omissão é a GRUClassifier do notebook 04 (~106 mil parâmetros).
    """

    def __init__(self, n_features=13, hidden=64, layers=2, dropout=0.3, cell="gru", n_classes=len(CLASSES)):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)   # (média + máximo) x 2 direções

    def forward(self, x):                            # x: (batch, 313, n_features)
        out, _ = self.rnn(x)                         # (batch, 313, 2 * hidden)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        return self.fc(self.dropout(pooled))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
