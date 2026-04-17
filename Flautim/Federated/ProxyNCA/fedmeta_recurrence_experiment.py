import copy
import gc
from collections import OrderedDict
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from flautim.pytorch.federated.Experiment import Experiment
from flwr.common.typing import NDArrays, Scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
from torch.cuda.amp import autocast

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


class RecurrenceExperiment(Experiment):
    def __init__(
        self,
        context,
        net: MetaMetricModel,
        dataset,
        client_name: int = 0,
        cfg: HFMetaMetricConfig = HFMetaMetricConfig(),
        **kwargs,
    ):
        super(RecurrenceExperiment, self).__init__(net, dataset, context, **kwargs)
        self.net = net
        self.client_name = int(client_name)
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.loss_fn = ProxyNCALoss(
            ProxyNCAConfig(
                temperature=cfg.temperature,
                normalize_embeddings=cfg.normalize_embeddings,
                normalize_proxies=cfg.normalize_proxies,
            )
        )

        self.adapted_backbone: nn.Module | None = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [v.cpu().numpy() for _, v in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    ################################
    # TRAINING FUNCTIONS ######
    ################################

    def train_model(
        self, model, adapted_backbone, train_loader, alpha, gradient_step=5
    ):
        """Train with minimal memory footprint."""
        model.train()
        total_loss, total_samples = (0.0, 0)

        for _ in range(gradient_step):
            for data, labels in train_loader:
                data, labels = (
                    data.float().to(self.device),
                    labels.long().to(self.device),
                )


                # Mixed precision forward
                with autocast(enabled=True):
                    emb = extract_embeddings(adapted_backbone, data)
                    loss_proxy = self.loss_fn(emb, model.proxies, labels)
                    loss = loss_proxy

                total_loss += float(loss.detach().cpu()) * labels.size(0)
                total_samples += labels.size(0)

                params = list(adapted_backbone.parameters())
                grads = torch.autograd.grad(
                    loss, 
                    params, 
                    retain_graph=False, 
                    create_graph=False, 
                    allow_unused=True
                )

                grads_backbone = grads[:-1]
                grads_delta = grads[-1]

                # Update parameters without tracking gradients
                with torch.no_grad():
                    for param, grad_ in zip(params, grads_backbone):
                        if grad_ is not None:
                            param.add_(grad_, alpha=-float(alpha))

                # Clear tensors to avoid graph accumulation
                del (
                    data,
                    labels,
                    emb,
                    loss,
                    loss_proxy,
                    loss_reg,
                    grads,
                    grads_backbone,
                    grads_delta,
                )
                torch.cuda.empty_cache()

            torch.cuda.synchronize()

        return total_loss / max(1, total_samples)

    def compute_query_loss(self, model, adapted_backbone, query_loader):

        model.proxies.requires_grad_(True)

        total_samples = 0
        loss_sum = 0.0

        named_backbone = list(adapted_backbone.named_parameters())
        names = [f"backbone.{n}" for n, _ in named_backbone] + ["proxies"]
        params = [p for _, p in named_backbone] + [model.proxies]

        for data, labels in query_loader:
            data = data.float().to(self.device)
            labels = labels.long().to(self.device)

            batch_size = labels.size(0)
            total_samples += batch_size

            with torch.cuda.amp.autocast(enabled=True):
                emb = extract_embeddings(adapted_backbone, data)
                loss = self.loss_fn(emb, model.proxies, labels)

            # IMPORTANT: keep tensor, not .item()
            loss_sum += loss * batch_size

            del data, labels, emb, loss

        loss_sum = loss_sum / total_samples

        grads = torch.autograd.grad(
            loss_sum, params, create_graph=False, allow_unused=True
        )

        avg_grads = {
            name: g.detach().cpu().numpy()
            for name, g in zip(names, grads)
            if g is not None
        }

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return avg_grads, total_samples, loss_sum.item()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.log(
            f"{self.client_name} - FedMeta client training started",
            object_id=self.id,
        )
        self.epoch_fl = config["server_round"]

        # === Load global model ===
        self.set_parameters(parameters)

        # === Clear any old adapted model ===
        if hasattr(self, "adapted_backbone"):
            del self.adapted_backbone
            torch.cuda.empty_cache()
            gc.collect()

        self.adapted_backbone = copy.deepcopy(self.net.backbone).to(self.device)
        for p in self.adapted_backbone.parameters():
            p.requires_grad_(True)

        # === Reload dataloaders ===
        support_loader, query_loader = self.dataset.dataloader(validation=False)

        # === Adapt model and calculate grads ===
        loss = self.train_model(
            self.net,
            self.adapted_backbone,
            support_loader,
            self.cfg.alpha,
            self.cfg.inner_epochs,
        )

        grads, num_examples, query_loss = self.compute_query_loss(
            self.net, self.adapted_backbone, query_loader
        )

        file_path = f"gradients_{self.client_name}.npz"
        np.savez(file_path, **grads)

        # === Clear memory ===
        del self.adapted_backbone, grads
        torch.cuda.empty_cache()
        gc.collect()

        self.log(
            f"{self.client_name} - FedMeta client finished",
            object_id=self.id,
        )

        return (
            self.get_parameters({}),
            num_examples,
            {"loss": loss, "query_loss": query_loss, "grads_file": self.client_name},
        )

    ################################
    # TESTING FUNCTIONS ######
    ################################

    def validation_loop(self, data_loaders):
        support_loader, query_loader = data_loaders

        # ---- Inner loop adaptation ----
        self.adapted_backbone = copy.deepcopy(self.net.backbone).to(self.device)

        self.train_model(
            self.net,
            self.adapted_backbone,
            support_loader,
            self.cfg.alpha,
            self.cfg.inner_epochs,
        )

        self.adapted_backbone.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in query_loader:
                X = X.float().to(self.device)
                y = y.long().to(self.device)
                batch_size = y.size(0)

                with autocast(enabled=True):
                    emb = extract_embeddings(self.adapted_backbone, X)
                    loss = self.loss_fn(emb, self.net.proxies, y)

                # logits for prediction (mirror loss settings)
                if self.cfg.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                P = self.net.proxies
                if self.cfg.normalize_proxies:
                    P = torch.nn.functional.normalize(P, p=2, dim=-1)

                with torch.cuda.amp.autocast(enabled=False):
                    emb_fp32 = emb.float()
                    P_fp32 = P.float()
                    logits = (emb_fp32 @ P_fp32.t()) / float(self.cfg.temperature)

                pred = torch.argmax(logits, dim=1)
                total_loss += float(loss.detach().cpu()) * batch_size
                total_samples += batch_size
                all_preds.extend(pred.detach().cpu().tolist())
                all_labels.extend(y.detach().cpu().tolist())

                del (X, y, emb, loss, logits, pred)
                torch.cuda.empty_cache()

        loss_avg = total_loss / max(1, total_samples)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return (
            loss_avg,
            total_samples,
            {
                "TEST_ACC": balanced_acc,
                "TEST_PREC": precision,
                "TEST_F1": f1,
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        return_dic = {}

        self.log(
            f"{self.client_name} - FedMeta Model evaluation started",
            object="experiment_evaluate",
            object_id=self.id,
        )

        self.set_parameters(parameters)
        self.epoch_fl = config["server_round"]

        loss, num_examples, values_metrics_validation = self.validation_loop(
            self.dataset.dataloader(validation=True)
        )
        values_metrics_validation["LOSS"] = loss

        for name in values_metrics_validation:
            self.measures.log(
                self,
                name,
                values_metrics_validation[name],
                validation=True,
                epoch=self.epoch_fl,
            )
            return_dic[name] = float(values_metrics_validation[name])

        self.net.save()
        self.log(
            f"{self.client_name} - evaluation finished",
            details="",
            object="experiment_evaluate",
            object_id=self.id,
        )

        torch.cuda.empty_cache()
        return (
            float(loss),
            num_examples,
            return_dic,
        )
