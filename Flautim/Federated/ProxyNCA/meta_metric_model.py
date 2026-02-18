from __future__ import annotations

from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn


def extract_embeddings(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Preferência:
      1) se existir forward_embedding(x), usa
      2) senão, tenta padrões (Tensor / tuple / dict)
    """
    if hasattr(backbone, "forward_embedding") and callable(getattr(backbone, "forward_embedding")):
        return backbone.forward_embedding(x)

    out = backbone(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for k in ("embeddings", "embedding", "feat", "features", "z"):
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
    raise TypeError(f"Backbone output not supported: {type(out)}")


class MetaMetricModel(nn.Module):
    """
    backbone: embedding extractor
    proxies:  nn.Parameter [C, D] (entra no state_dict)
    """

    def __init__(self, backbone: nn.Module, num_classes: int, emb_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)
        self.proxies = nn.Parameter(init_scale * torch.randn(self.num_classes, self.emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_embeddings(self.backbone, x)

    def get_parameters(self) -> List:
        return [v.detach().cpu().numpy() for _, v in self.state_dict().items()]

    def set_parameters(self, parameters: List) -> None:
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

