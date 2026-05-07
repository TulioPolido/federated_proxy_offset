# proxy_nca_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProxyNCAConfig:
    temperature: float = 0.1
    normalize_embeddings: bool = True
    normalize_proxies: bool = True


class ProxyNCALoss(nn.Module):
    """
    Proxy-NCA style metric learning implemented as:
        logits = (E @ P^T) / T
        loss  = CE(logits, y)

    where E are embeddings and P are class proxies.
    Optional L2-normalization for E and/or P.
    """

    def __init__(self, cfg: ProxyNCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, D]
        proxies: torch.Tensor,     # [C, D]
        targets: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        if self.cfg.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        if self.cfg.normalize_proxies:
            proxies = F.normalize(proxies, p=2, dim=-1)

        logits = embeddings @ proxies.t()
        logits = logits / float(self.cfg.temperature)
        return F.cross_entropy(logits, targets)

