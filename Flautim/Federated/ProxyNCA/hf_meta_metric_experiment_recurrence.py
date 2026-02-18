from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score

from flwr.common.typing import NDArrays, Scalar
from flautim.pytorch.federated.Experiment import Experiment

from proxy_nca_loss import ProxyNCALoss, ProxyNCAConfig
from meta_metric_model import MetaMetricModel, extract_embeddings


@dataclass
class HFMetaMetricConfig:
    alpha: float = 1e-4              # inner LR
    inner_epochs: int = 1            # epochs over support loader
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
        client_name: int = 0,
        cfg: HFMetaMetricConfig = HFMetaMetricConfig(),
        **kwargs,
    ):
        super().__init__(model, dataset, context, **kwargs)
        self.client_name = int(client_name)
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = ProxyNCALoss(
            ProxyNCAConfig(
                temperature=cfg.temperature,
                normalize_embeddings=cfg.normalize_embeddings,
                normalize_proxies=cfg.normalize_proxies,
            )
        )

        self.adapted_backbone: nn.Module | None = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # -------- Inner loop (support) --------
    def train_model_with_offsets(
        self,
        adapted_backbone: nn.Module,
        support_loader,
        deltaP: torch.Tensor,  # [C, D], requires_grad=True
    ) -> float:
        adapted_backbone.train()
        total_loss, total_samples = 0.0, 0

        for _ in range(int(self.cfg.inner_epochs)):
            for x, y in support_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                bs = y.size(0)

                proxies_eff = self.model.proxies + deltaP  # P_client

                with autocast(enabled=bool(self.cfg.use_amp and self.device.type == "cuda")):
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
                    allow_unused=True,   # <-- FIX
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


                del x, y, emb, loss, loss_proxy, loss_reg, grads, grads_backbone, grad_delta

        return total_loss / max(1, total_samples)

    # -------- Query grads (meta gradient) --------
    def compute_query_grads(
        self,
        adapted_backbone: nn.Module,
        query_loader,
        deltaP: torch.Tensor,
    ) -> Tuple[List[np.ndarray], int, float]:
        adapted_backbone.eval()
        self.model.proxies.requires_grad_(True)

        # Em vez de params = list(adapted_backbone.parameters()) + [self.model.proxies]
        # vamos pegar NOMES compatíveis com o modelo global:
        named_backbone = list(adapted_backbone.named_parameters())
        names = [f"backbone.{n}" for n, _ in named_backbone] + ["proxies"]
        params = [p for _, p in named_backbone] + [self.model.proxies]

        grads_sum = {name: torch.zeros_like(param, device="cpu") for name, param in zip(names, params)}


        total_samples = 0
        total_loss = 0.0

        proxies_eff = self.model.proxies + deltaP.detach()  # IMPORTANT: no grad into ΔP

        for x, y in query_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = y.size(0)
            total_samples += bs

            with autocast(enabled=bool(self.cfg.use_amp and self.device.type == "cuda")):
                emb = extract_embeddings(adapted_backbone, x)
                loss = self.loss_fn(emb, proxies_eff, y)

            total_loss += float(loss.detach().cpu()) * bs

            grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)

            for name, g in zip(names, grads):
                if g is None:
                    continue
                grads_sum[name].add_(g.detach().cpu() * bs)



            del x, y, emb, loss, grads
            
        avg_grads = {k: (v / max(1, total_samples)).numpy() for k, v in grads_sum.items()}
        return avg_grads, int(total_samples), total_loss / max(1, total_samples)



    # -------- Flower fit() --------
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.log(f"{self.client_name} - HFMetaMetric client training started", details="", object="", object_id=self.id)
        self.epoch_fl = int(config.get("server_round", 0))

        # Load global
        self.set_parameters(parameters)

        # Copy backbone for adaptation
        if self.adapted_backbone is not None:
            del self.adapted_backbone
        self.adapted_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        
        self.adapted_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        for p in self.adapted_backbone.parameters():
            p.requires_grad_(True)


        # ΔP init zero each round
        deltaP = torch.zeros_like(self.model.proxies, device=self.device, requires_grad=True)

        # RecurrenceDatasetnew returns [train_loader, test_loader]
        support_loader, query_loader = self.dataset.dataloader(validation=False)

        support_loss = self.train_model_with_offsets(self.adapted_backbone, support_loader, deltaP)
        query_grads, query_n, query_loss = self.compute_query_grads(self.adapted_backbone, query_loader, deltaP)

        grads_file = f"hfmm_{self.client_name}"
        np.savez(f"gradients_{grads_file}.npz", **query_grads)

        del deltaP

        self.log(f"{self.client_name} - HFMetaMetric client finished", details="", object="", object_id=self.id)

        return (
            self.get_parameters({}),
            int(query_n),  # IMPORTANT: weight by query size
            {"support_loss": float(support_loss), "query_loss": float(query_loss), "grads_file": grads_file},
        )

    # -------- Evaluation (adapt on support, test on query) --------
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.log(f"{self.client_name} - HFMetaMetric evaluation started", details="", object="experiment_evaluate", object_id=self.id)
        self.epoch_fl = int(config.get("server_round", 0))

        self.set_parameters(parameters)

        support_loader, query_loader = self.dataset.dataloader(validation=True)

        adapted_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        deltaP = torch.zeros_like(self.model.proxies, device=self.device, requires_grad=True)

        _ = self.train_model_with_offsets(adapted_backbone, support_loader, deltaP)

        adapted_backbone.eval()
        proxies_eff = (self.model.proxies + deltaP.detach())

        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss, total_samples = 0.0, 0

        for x, y in query_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = y.size(0)

            with autocast(enabled=bool(self.cfg.use_amp and self.device.type == "cuda")):
                emb = extract_embeddings(adapted_backbone, x)
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

        loss_avg = total_loss / max(1, total_samples)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        metrics = {
            "LOSS": float(loss_avg),
            "TEST_ACC": float(bal_acc),
            "TEST_PREC": float(prec),
            "TEST_F1": float(f1),
        }

        for k, v in metrics.items():
            try:
                self.measures.log(self, k, v, validation=True, epoch=self.epoch_fl)
            except Exception:
                pass

        self.log(f"{self.client_name} 