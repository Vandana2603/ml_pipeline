"""
training/model.py
Model definitions: MLP (primary), with extensible base class.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for classification.
    Supports variable depth, dropout, and BatchNorm.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        act_fn = act_map.get(activation, nn.ReLU)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "mlp",
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hidden_dims": self.hidden_dims,
        }


def build_model(config: Dict[str, Any], input_dim: int, n_classes: int) -> nn.Module:
    """Factory: build model from config dict."""
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "mlp")

    if model_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.3),
            activation=model_cfg.get("activation", "relu"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
