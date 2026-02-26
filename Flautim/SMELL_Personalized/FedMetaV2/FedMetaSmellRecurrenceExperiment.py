from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import numpy as np
import gc
import os


from flautim.pytorch.federated.Experiment import Experiment
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import random
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from sklearn.neighbors import KNeighborsClassifier
import flautim as fl
from flwr.common.typing import NDArrays, Scalar
import copy

from sklearn.decomposition import PCA

from meta_metric_model import MetaMetricModel, extract_embeddings
from smell_loss import (
    LearnablePrototypicalLossWithDiversityRegularization,
    PrototypicalCenterLoss,
)


###############################
# EXPERIMENT
###############################


class RecurrenceExperiment(Experiment):
    def __init__(
        self,
        model,
        dataset,
        context,
        client_name=0,
        lr=0.001,
        pre_train_lr=0.001,
        epochs=10,
        weight_decay=1e-4,
        lr_step=3,
        lr_gamma=0.5,
        num_classes=7,
        embedding_size=64,
        margin=1.0,
        # pre_train=False,
        **kwargs,
    ):
        super(RecurrenceExperiment, self).__init__(model, dataset, context, **kwargs)

        # self.pre_train = pre_train
        self.epochs = epochs
        self.num_classes = num_classes
        self.margin = margin
        self.embedding_size = embedding_size
        self.lr = lr
        self.pre_train_lr = pre_train_lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.client_name = client_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.adapted_model = copy.deepcopy(self.model).to(self.device)

        self.loss_fn = LearnablePrototypicalLossWithDiversityRegularization(
            embedding_dim=self.embedding_size,
            num_similar_prototypes=2,
            num_dissimilar_prototypes=2,
        ).to(self.device)

        self.pre_criterion = PrototypicalCenterLoss(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            margin=self.margin,
        ).to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    ################################
    # TRAINING FUNCTIONS ######
    ################################

    def generate_pair_indices(
        self,
        Y: torch.Tensor,
        N: int,
        allow_repeat: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build index pairs inside the *current batch* using labels Y.

        - If only one class is present, only similar pairs are produced.
        - If not enough unique pairs exist, we either cap N or (if allow_repeat=True)
        we sample with replacement to meet the target.
        - Returns empty tensors if no pairs can be formed.

        Returns:
        idx1 [P], idx2 [P], y [P]  (on same device as Y)
        """
        device = Y.device
        N = int(N)
        if N <= 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )

        # Map: label -> indices in this batch
        uniques = torch.unique(Y)
        label_to_idx = {
            int(lbl.item()): (Y == lbl).nonzero(as_tuple=True)[0] for lbl in uniques
        }

        # Count max unique similar/dissimilar pairs possible in this batch
        # similar: sum over labels of C(n_l, 2)
        # dissimilar: sum over l1<l2 of n_l1 * n_l2
        sizes = [idx.numel() for idx in label_to_idx.values()]
        max_similar = sum(max(0, n * (n - 1) // 2) for n in sizes)
        if len(sizes) >= 2:
            total = sum(sizes)
            sum_sq = sum(n * n for n in sizes)
            max_dissimilar = (total * total - sum_sq) // 2  # = sum_{i<j} n_i n_j
        else:
            max_dissimilar = 0

        # target split (half/half), then cap by feasibility
        target_sim = N // 2
        target_dis = N - target_sim

        # If only one class exists, force dis=0 and push all to sim
        if len(sizes) < 2:
            target_dis = 0
            target_sim = min(N, max_similar) if not allow_repeat else N

        # Cap by unique counts if not allowing repeats
        if not allow_repeat:
            target_sim = min(target_sim, max_similar)
            target_dis = min(target_dis, max_dissimilar)

        # If both are zero and repeats not allowed (e.g., tiny batch), return empties
        if target_sim <= 0 and target_dis <= 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )

        idx1_list, idx2_list, y_list = [], [], []

        # --- Similar pairs ---
        if target_sim > 0:
            # For each label with >=2 samples, create pairs
            # We’ll generate with replacement if needed.
            for _ in range(target_sim):
                # pick a label with n>=2
                valid_labels = [k for k, v in label_to_idx.items() if v.numel() >= 2]
                if not valid_labels:
                    break
                lb = random.choice(valid_labels)
                v = label_to_idx[lb]
                # sample two distinct indices; if allow_repeat, we just sample again
                if v.numel() >= 2:
                    # draw two different positions
                    i1, i2 = (
                        random.sample(range(v.numel()), 2) if v.numel() >= 2 else (0, 0)
                    )
                    idx1_list.append(v[i1].item())
                    idx2_list.append(v[i2].item())
                    y_list.append(1.0)

        # --- Dissimilar pairs ---
        if target_dis > 0 and len(label_to_idx) >= 2:
            labels_list = list(label_to_idx.keys())
            for _ in range(target_dis):
                if len(labels_list) < 2:
                    break
                # choose two different labels; with repeat we can reuse previously used combinations
                l1, l2 = random.sample(labels_list, 2)
                v1, v2 = label_to_idx[l1], label_to_idx[l2]
                if v1.numel() == 0 or v2.numel() == 0:
                    continue
                i1 = v1[random.randrange(v1.numel())].item()
                i2 = v2[random.randrange(v2.numel())].item()
                idx1_list.append(i1)
                idx2_list.append(i2)
                y_list.append(0.0)

        # Pack to tensors on the right device
        if len(y_list) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )

        idx1 = torch.tensor(idx1_list, dtype=torch.long, device=device)
        idx2 = torch.tensor(idx2_list, dtype=torch.long, device=device)
        y = torch.tensor(y_list, dtype=torch.float32, device=device)
        return idx1, idx2, y

    def pre_train_model(self, model, dataloader, num_epochs=1, alpha=0.001):

        model.train()
        self.pre_criterion.train()

        # Combine parameters from model + pre_criterion
        train_params = list(model.parameters()) + list(self.pre_criterion.parameters())

        for epoch in range(num_epochs):
            running_loss = 0.0

            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward
                embeddings = model(batch_data.float())
                loss = self.pre_criterion(embeddings, batch_labels)

                running_loss += loss.item()

                # Compute gradients
                grads = torch.autograd.grad(
                    loss, train_params, create_graph=False, allow_unused=True
                )

                # Manual SGD update
                with torch.no_grad():
                    for p, g in zip(train_params, grads):
                        if g is not None:
                            p -= alpha * g

                # Cleanup
                del embeddings, loss, batch_data, batch_labels, grads
                torch.cuda.empty_cache()

            torch.cuda.synchronize()

        return running_loss / max(1, len(dataloader))

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

                # COMO ISSO SERÁ USADO COM O SMELL??
                proxies_eff = self.model.proxies + deltaP  # P_client

                with autocast(
                    enabled=bool(self.cfg.use_amp and self.device.type == "cuda")
                ):
                    idx1, idx2, y = self.generate_pair_indices(y, N=200)
                    emb1 = extract_embeddings(adapted_backbone, x[idx1])
                    emb2 = extract_embeddings(adapted_backbone, x[idx2])
                    loss_proxy = self.loss_fn(emb1, emb2, y)
                    loss_reg = 0  # self.cfg.lambda_offset * deltaP.pow(2).sum()
                    loss = loss_proxy  # + loss_reg

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
                    emb1,
                    emb2,
                    loss,
                    loss_proxy,
                    # loss_reg,
                    grads,
                    grads_backbone,
                    grad_delta,
                )

        return total_loss / max(1, total_samples)

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

        grads_sum = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in zip(names, params)
        }

        total_samples = 0
        total_loss = 0.0

        proxies_eff = self.model.proxies + deltaP.detach()  # IMPORTANT: no grad into ΔP

        for x, y in query_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            bs = y.size(0)
            total_samples += bs

            with autocast(
                enabled=bool(self.cfg.use_amp and self.device.type == "cuda")
            ):
                idx1, idx2, y = self.generate_pair_indices(y, N=200)
                emb1 = extract_embeddings(adapted_backbone, x[idx1])
                emb2 = extract_embeddings(adapted_backbone, x[idx2])
                loss = self.loss_fn(emb1, emb2, y)

            total_loss += float(loss.detach().cpu()) * bs

            grads = torch.autograd.grad(
                loss, params, create_graph=False, retain_graph=False, allow_unused=True
            )

            for name, g in zip(names, grads):
                if g is None:
                    continue
                grads_sum[name].add_(g.detach().cpu() * bs)

            del x, y, emb1, emb2, idx1, idx2, loss, grads

        avg_grads = {
            k: (v / max(1, total_samples)).numpy() for k, v in grads_sum.items()
        }
        return avg_grads, int(total_samples), total_loss / max(1, total_samples)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        self.log(
            f"{self.client_name} - HFMetaMetric client training started",
            details="",
            object="",
            object_id=self.id,
        )
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
        deltaP = torch.zeros_like(
            self.model.proxies, device=self.device, requires_grad=True
        )

        # RecurrenceDatasetnew returns [train_loader, test_loader]
        support_loader, query_loader = self.dataset.dataloader(validation=False)

        support_loss = self.train_model_with_offsets(
            self.adapted_backbone, support_loader, deltaP
        )
        query_grads, query_n, query_loss = self.compute_query_grads(
            self.adapted_backbone, query_loader, deltaP
        )

        grads_file = f"hfmm_{self.client_name}"
        np.savez(f"gradients_{grads_file}.npz", **query_grads)

        del deltaP

        self.log(
            f"{self.client_name} - HFMetaMetric client finished",
            details="",
            object="",
            object_id=self.id,
        )

        return (
            self.get_parameters({}),
            int(query_n),  # IMPORTANT: weight by query size
            {
                "support_loss": float(support_loss),
                "query_loss": float(query_loss),
                "grads_file": grads_file,
            },
        )

    ################################
    # TESTING  FUNCTIONS ######
    ################################

    def knn_in_embedding_space(
        self, model, train_loader, test_loader, k=1, num_samples_for_pairs=200
    ):

        # collect train embeddings – use inference_mode and float16 on CPU
        train_emb_list, train_lbl_list = [], []
        with torch.inference_mode():
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device, non_blocking=True).float()
                emb = model(batch_data)
                train_emb_list.append(
                    emb.detach().cpu().to(torch.float16)
                )  # 2x smaller
                train_lbl_list.append(batch_labels.cpu())
                del emb, batch_data, batch_labels

        train_embeddings = torch.cat(train_emb_list).numpy()  # float16 numpy
        train_labels = torch.cat(train_lbl_list).numpy()

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings, train_labels)

        all_preds, all_labels = [], []
        running_loss, total_batches = 0.0, 0

        with torch.inference_mode():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device).float()

                # compute embeddings on GPU once
                emb_gpu = model(batch_data)  # [B, D]
                emb_cpu = emb_gpu.detach().cpu().to(torch.float16)  # for KNN

                preds = knn.predict(emb_cpu.numpy())
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())

                # build pairs on GPU
                idx1, idx2, y = self.generate_pair_indices(
                    batch_labels.to(self.device), num_samples_for_pairs
                )

                # match train style: direct indexing, no index_select
                e1 = emb_gpu[idx1]
                e2 = emb_gpu[idx2]
                loss = self.loss_fn(e1, e2, y)

                running_loss += loss.item()
                total_batches += 1

                del (
                    e1,
                    e2,
                    idx1,
                    idx2,
                    y,
                    emb_gpu,
                    emb_cpu,
                    batch_data,
                    batch_labels,
                )
                torch.cuda.empty_cache()

        self.log(
            f"{self.client_name} - Evaluate(): FedMeta Model evaluation started",
            object="experiment_evaluate",
            object_id=self.id,
        )

        avg_loss = running_loss / max(1, total_batches)
        y_pred = torch.tensor(all_preds)
        y_label = torch.tensor(all_labels)
        return avg_loss, y_pred, y_label

    def test(self, model, trainloader, testloader):
        loss = 0.0
        model.to(self.device)
        model.eval()
        self.loss_fn.train()

        accuracy = MulticlassAccuracy(
            num_classes=self.num_classes, average=None  # "macro"
        ).to(self.device)

        precision = MulticlassPrecision(
            num_classes=self.num_classes, average="macro"
        ).to(self.device)

        recall = MulticlassRecall(num_classes=self.num_classes, average="macro").to(
            self.device
        )

        f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro").to(
            self.device
        )

        conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes).to(
            self.device
        )

        loss, predicted, label = self.knn_in_embedding_space(
            model, trainloader, testloader
        )

        predicted = torch.tensor(predicted).to(self.device)
        label = torch.tensor(label).to(self.device)

        accuracy.update(predicted, label)
        precision.update(predicted, label)
        recall.update(predicted, label)
        f1.update(predicted, label)
        conf_matrix.update(predicted, label)

        acc = accuracy.compute().mean().item()  # macro average manually
        acc_per_class = accuracy.compute().cpu().numpy().tolist()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()
        conf = conf_matrix.compute().cpu().numpy()

        return float(loss), acc, acc_per_class, prec, rec, f1_score, conf

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        self.log(
            f"{self.client_name} - HFMetaMetric evaluation started",
            details="",
            object="experiment_evaluate",
            object_id=self.id,
        )
        self.epoch_fl = int(config.get("server_round", 0))

        self.set_parameters(parameters)

        support_loader, query_loader = self.dataset.dataloader(validation=True)

        adapted_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        deltaP = torch.zeros_like(
            self.model.proxies, device=self.device, requires_grad=True
        )

        _ = self.train_model_with_offsets(adapted_backbone, support_loader, deltaP)

        # ---- Evaluate adapted model on query set ----
        loss, acc, acc_per_class, prec, rec, f1, conf = self.test(
            adapted_backbone, support_loader, query_loader
        )

        values_metrics_validation = {
            "LOSS": float(loss),
            "TEST_ACC": float(acc),
            "TEST_PREC": float(prec),
            "TEST_F1": float(f1),
        }

        return_dic = {}
        for name in values_metrics_validation:
            self.measures.log(
                self,
                name,
                values_metrics_validation[name],
                validation=True,
                epoch=self.epoch_fl,
            )
            return_dic[name] = float(values_metrics_validation[name])

        self.model.save()
        self.log(
            f"{self.client_name} - evaluation finished",
            details="",
            object="experiment_evaluate",
            object_id=self.id,
        )

        torch.cuda.empty_cache()
        return (
            float(loss),
            len(self.dataset.dataloader(validation=True)),
            return_dic,
        )
