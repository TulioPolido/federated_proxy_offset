from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score

from flautim.pytorch.centralized.Experiment import Experiment

from proxy_nca_loss import ProxyNCALoss, ProxyNCAConfig
from meta_metric_model import MetaMetricModel, extract_embeddings


@dataclass
class HFMetaMetricConfig:
    alpha: float = 1e-4  # inner LR
    inner_epochs: int = 1  # epochs over support loader
    lambda_offset: float = 1e-3
    temperature: float = 0.1
    normalize_embeddings: bool = True
    normalize_proxies: bool = True
    use_amp: bool = True


class HFMetaMetricRecurrenceExperiment(Experiment):
    """
    Plug-and-play no seu framework (extends flautim Experiment).

    Dataset: RecurrenceDatasetnew (dataloader() retorna [support_loader, query_loader]).
    """

    def __init__(
        self,
        model: MetaMetricModel,
        dataset,
        context,
        epochs=10,
        client_name: int = 0,
        cfg: HFMetaMetricConfig = HFMetaMetricConfig(),
        **kwargs,
    ):
        super().__init__(model, dataset, context, **kwargs)
        self.client_name = int(client_name)
        self.cfg = cfg
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = ProxyNCALoss(
            ProxyNCAConfig(
                temperature=cfg.temperature,
                normalize_embeddings=cfg.normalize_embeddings,
                normalize_proxies=cfg.normalize_proxies,
            )
        )

    # -------- Inner loop (support) --------
    def train_model_with_offsets(
        self, adapted_backbone: nn.Module, support_loader
    ) -> float:

        # ΔP init zero each round
        deltaP = torch.zeros_like(
            self.model.proxies, device=self.device, requires_grad=True
        )

        adapted_backbone.train()
        total_loss, total_samples = 0.0, 0

        for x, y in support_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = y.size(0)

            proxies_eff = self.model.proxies + deltaP  # P_client

            with autocast(
                enabled=bool(self.cfg.use_amp and self.device.type == "cuda")
            ):
                emb = extract_embeddings(adapted_backbone, x)
                loss_proxy = self.loss_fn(emb, proxies_eff, y)
                loss_reg = self.cfg.lambda_offset * deltaP.pow(2).sum()
                loss = loss_proxy + loss_reg

            total_loss += float(loss.detach().cpu()) * bs
            total_samples += bs

            params = list(adapted_backbone.parameters())
            grads = torch.autograd.grad(
                loss,
                params + [deltaP],
                create_graph=False,
                retain_graph=False,
                allow_unused=True,  # <-- FIX
            )

            grads_backbone = grads[:-1]
            grad_delta = grads[-1]

            with torch.no_grad():
                for p, g in zip(params, grads_backbone):
                    if g is None:
                        continue
                    p.add_(g, alpha=-float(self.cfg.alpha))

                # Se deltaP não apareceu no grafo por algum motivo, não atualiza
                if grad_delta is not None:
                    deltaP.add_(grad_delta, alpha=-float(self.cfg.alpha))

            del (
                x,
                y,
                emb,
                loss,
                loss_proxy,
                loss_reg,
                grads,
                grads_backbone,
                grad_delta,
            )

        del deltaP

        return total_loss / max(1, total_samples)

    def training_loop(self, data_loader):
        train_loader, _ = data_loader
        loss_avg = self.train_model_with_offsets(
            adapted_backbone=self.model.backbone, support_loader=train_loader
        )
        return float(loss_avg), {}

    def validation_loop(self, data_loader):
        deltaP = torch.zeros_like(
            self.model.proxies, device=self.device, requires_grad=True
        )
        _, test_loader = data_loader
        self.model.backbone.eval()
        proxies_eff = self.model.proxies + deltaP.detach()

        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss, total_samples = 0.0, 0

        for x, y in test_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = y.size(0)

            with autocast(
                enabled=bool(self.cfg.use_amp and self.device.type == "cuda")
            ):
                emb = extract_embeddings(self.model.backbone, x)
                loss = self.loss_fn(emb, proxies_eff, y)

            # logits for prediction (mirror loss settings)
            if self.cfg.normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            P = proxies_eff
            if self.cfg.normalize_proxies:
                P = torch.nn.functional.normalize(P, p=2, dim=-1)

            with torch.cuda.amp.autocast(enabled=False):
                emb_fp32 = emb.float()
                P_fp32 = P.float()
                logits = (emb_fp32 @ P_fp32.t()) / float(self.cfg.temperature)

            pred = torch.argmax(logits, dim=1)

            total_loss += float(loss.detach().cpu()) * bs
            total_samples += bs
            all_preds.extend(pred.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

            del x, y, emb, loss, logits, pred

        loss = total_loss / max(1, total_samples)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        prec = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return float(loss), {
            "LOSS": float(loss),
            "TEST_ACC": float(bal_acc),
            "TEST_PREC": float(prec),
            "TEST_F1": float(f1),
        }
