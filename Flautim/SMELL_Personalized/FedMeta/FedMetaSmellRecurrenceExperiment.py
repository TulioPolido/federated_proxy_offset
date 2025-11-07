from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import numpy as np
import gc


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


###############################
# PROTOTIPICAL CENTER LOSS
###############################


class PrototypicalCenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=2.0):
        """
        Prototypical center loss to minimize intra-class distance and maximize inter-class distance.

        :param num_classes: The number of distinct classes.
        :param embedding_size: The dimension of the embedding space.
        :param margin: A margin to separate class centers.
        """
        super(PrototypicalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin

        # Initialize class prototypes (centers) as learnable parameters
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, embedding_size), requires_grad=True
        )  # create a parameter tensor of shape (num_classes, embedding_size)
        # The function is to make each 'prototype' as a possible center of one class

    def forward(self, embeddings, labels):
        """
        Compute prototypical center loss.

        :param embeddings: The input embeddings of shape [batch_size, embedding_dim].
        :param labels: The corresponding labels of shape [batch_size].
        :return: The prototypical center loss.
        """
        # batch_size = embeddings.size(0)

        # Compute distance to the corresponding class prototype (center)
        dist_to_center = torch.norm(embeddings - self.prototypes[labels], dim=1)

        # Intra-class loss (minimize distance to the correct class prototype)
        loss_intra = torch.mean(torch.pow(dist_to_center, 2))

        # Optionally, add inter-class loss (encourage prototypes to be apart)
        inter_class_distances = torch.cdist(self.prototypes, self.prototypes)
        inter_class_distances = (
            inter_class_distances
            + torch.eye(self.num_classes, device=embeddings.device) * 1e12
        )  # Prevent self-comparison

        # Apply margin: penalize if prototypes are closer than the margin
        loss_inter = torch.mean(F.relu(self.margin - inter_class_distances))

        total_loss = loss_intra + loss_inter
        return total_loss


class LearnablePrototypicalLossWithDiversityRegularization(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_similar_prototypes,
        num_dissimilar_prototypes,
        margin=1.0,
        lambda_reg=0.01,
    ):
        """
        Initialize the Prototypical Loss function with diversity regularization.
        paper : A New Similarity Space Tailored for Supervised Deep Metric Learning

        :param embedding_dim: The dimension of the input embeddings.
        :param num_similar_prototypes: Number of prototypical vectors for similar pairs.
        :param num_dissimilar_prototypes: Number of prototypical vectors for dissimilar pairs.
        :param margin: The minimum distance margin between prototypes.
        :param lambda_reg: Regularization strength.
        """
        super(LearnablePrototypicalLossWithDiversityRegularization, self).__init__()

        # the first similar prototype is all zeros, the rest is all random
        self.similar_prototypes = nn.Parameter(
            torch.cat(
                (
                    torch.zeros(1, embedding_dim),
                    torch.randn(num_similar_prototypes - 1, embedding_dim),
                ),
                dim=0,
            ),
            requires_grad=True,
        )

        # starts the dissimilar prototypes as all random + 2.0
        self.dissimilar_prototypes = nn.Parameter(
            torch.randn(num_dissimilar_prototypes, embedding_dim) + 2.0,
            requires_grad=True,
        )
        self.margin = margin
        self.lambda_reg = lambda_reg

    def toggle_requires_grad(self, index=0, requires_grad_value=True):
        """
        Function to toggle requires_grad for specific vectors in self.similar_prototypes
        """
        # Detach the tensor first to prevent in-place modification errors
        with torch.no_grad():
            # Set the requires_grad for a specific vector at index
            self.similar_prototypes.data[index] = self.similar_prototypes.data[
                index
            ].requires_grad_(requires_grad_value)
            self.dissimilar_prototypes.data[index] = self.dissimilar_prototypes.data[
                index
            ].requires_grad_(requires_grad_value)

    def diversity_regularization(self, prototypes):
        """
        Computes the diversity regularization term to prevent prototypes from collapsing.
        Encourages all prototypes to be at least `margin` distance apart.

        :param prototypes: The prototypical vectors (similar or dissimilar).
        :return: A scalar representing the regularization loss.
        """
        num_prototypes = prototypes.size(0)
        reg_loss = 0.0

        for i in range(num_prototypes):
            for j in range(i + 1, num_prototypes):
                # Compute the Euclidean distance between prototype i and j
                dist = torch.norm(prototypes[i] - prototypes[j])

                # Penalize if the distance is less than the margin
                if dist < self.margin:
                    reg_loss += torch.pow(self.margin - dist, 2)

        return reg_loss

    def forward(self, x1, x2, label):
        # Compute the absolute difference vector s_ij
        s_ij = torch.abs(x1 - x2)

        # Expand s_ij to compute distance to all prototypes
        s_ij = s_ij.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        # Compute distances to all similar prototypes
        dist_to_similar_prototypes = torch.cdist(
            s_ij, self.similar_prototypes.unsqueeze(0), p=2
        ).squeeze(1)
        dist_to_dissimilar_prototypes = torch.cdist(
            s_ij, self.dissimilar_prototypes.unsqueeze(0), p=2
        ).squeeze(1)

        # Compute similarity scores using the given formula
        similarity_similar = torch.pow(1 + torch.pow(dist_to_similar_prototypes, 2), -1)
        similarity_dissimilar = torch.pow(
            1 + torch.pow(dist_to_dissimilar_prototypes, 2), -1
        )

        # Normalization step: Sum of similarities for each prototype set
        denom_similar = torch.sum(similarity_similar, dim=1, keepdim=True)
        denom_dissimilar = torch.sum(similarity_dissimilar, dim=1, keepdim=True)

        # Normalize similarity scores
        similarity_similar_normalized = similarity_similar / (
            denom_similar + denom_dissimilar
        )
        similarity_dissimilar_normalized = similarity_dissimilar / (
            denom_similar + denom_dissimilar
        )

        # Calculate the loss:
        # If label == 1 (similar), minimize distance to similar prototypes
        # If label == 0 (dissimilar), minimize distance to dissimilar prototypes
        loss = -torch.mean(
            label * torch.log(torch.sum(similarity_similar_normalized, dim=1))
            + (1 - label)
            * torch.log(torch.sum(similarity_dissimilar_normalized, dim=1))
        )

        total_loss = loss

        # PORQUE ESTA COMENTADO?
        # Diversity regularization to avoid prototype collapse
        # diversity_loss_similar = self.diversity_regularization(self.similar_prototypes)
        # diversity_loss_dissimilar = self.diversity_regularization(self.dissimilar_prototypes)

        # Add diversity regularization to the total loss
        # total_loss += self.lambda_reg * (diversity_loss_similar + diversity_loss_dissimilar)

        return total_loss

    def compute_norms(self):
        # Compute norms for the similar prototypes
        similar_norms = torch.norm(
            self.similar_prototypes, dim=1
        )  # Compute L2 norm for each vector

        # Compute norms for the dissimilar prototypes
        dissimilar_norms = torch.norm(
            self.dissimilar_prototypes, dim=1
        )  # Compute L2 norm for each vector

        return similar_norms, dissimilar_norms


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

        self.criterion = LearnablePrototypicalLossWithDiversityRegularization(
            embedding_dim=self.embedding_size,
            num_similar_prototypes=2,
            num_dissimilar_prototypes=2,
        ).to(self.device)
        # self.optimizer1 = torch.optim.Adam(
        #     list(self.criterion.parameters()), lr=self.lr
        # )
        # self.optimizer2 = torch.torch.optim.SGD(
        #     [{"params": self.model.parameters(), "lr": self.lr}]
        #     + [{"params": self.criterion.parameters(), "lr": self.lr}]
        # )
        self.pre_criterion = PrototypicalCenterLoss(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            margin=self.margin,
        ).to(self.device)
        # self.pre_optimizer = torch.optim.Adam(
        #     list(model.parameters()) + list(self.pre_criterion.parameters()),
        #     lr=self.pre_train_lr,
        # )

        # self.scheduler = StepLR(self.optimizer1, step_size=lr_step, gamma=lr_gamma)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
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
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))

        # Map: label -> indices in this batch
        uniques = torch.unique(Y)
        label_to_idx = {int(lbl.item()): (Y == lbl).nonzero(as_tuple=True)[0] for lbl in uniques}

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
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))

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
                    i1, i2 = random.sample(range(v.numel()), 2) if v.numel() >= 2 else (0, 0)
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
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))

        idx1 = torch.tensor(idx1_list, dtype=torch.long, device=device)
        idx2 = torch.tensor(idx2_list, dtype=torch.long, device=device)
        y    = torch.tensor(y_list,    dtype=torch.float32, device=device)
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


    def train_model(self, model, train_loader, alpha, num_epochs=1, only_markers=False):
        model.train()
        self.criterion.train()  # criterion has learnable params

        # Choose params based on flag
        if only_markers:
            train_params = list(self.criterion.parameters())
        else:
            train_params = list(model.parameters()) + list(self.criterion.parameters())

        for epoch in range(num_epochs):
            total_loss, total_samples = 0.0, 0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device, non_blocking=True).float()
                batch_labels = batch_labels.to(self.device, non_blocking=True)

                with autocast(enabled=True):
                    idx1, idx2, y = self.generate_pair_indices(batch_labels, N=200)
                    e1 = model(batch_data[idx1])
                    e2 = model(batch_data[idx2])
                    loss = self.criterion(e1, e2, y)

                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

                # Compute gradients ONLY for chosen params
                grads = torch.autograd.grad(
                    loss, train_params, create_graph=False, allow_unused=True
                )

                # Manual SGD step
                with torch.no_grad():
                    for param, grad_ in zip(train_params, grads):
                        if grad_ is not None:
                            param -= alpha * grad_

                del e1, e2, idx1, idx2, y, grads, batch_data, batch_labels, loss
                torch.cuda.empty_cache()

            torch.cuda.synchronize()

        return total_loss / max(1, total_samples)



    def compute_query_loss(self, model, query_loader, num_samples_for_pairs=200):
        model.eval()
        self.criterion.eval()
        grads_sum = [torch.zeros_like(p, device="cpu") for p in model.parameters()]
        total_samples = 0

        for batch_data, batch_labels in query_loader:
            batch_data = batch_data.to(self.device, non_blocking=True).float()
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            batch_size = batch_labels.size(0)
            total_samples += batch_size

            idx1, idx2, y = self.generate_pair_indices(batch_labels, num_samples_for_pairs)
            with autocast(enabled=True):
                embeddings = model(batch_data)  # [B, D]
                e1 = embeddings.index_select(0, idx1)
                e2 = embeddings.index_select(0, idx2)
                loss = self.criterion(e1, e2, y)

            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)

            for i, g in enumerate(grads):
                if g is not None:
                    grads_sum[i] += g.detach().cpu() * batch_size

            del embeddings, e1, e2, idx1, idx2, y, loss, grads, batch_data, batch_labels
            torch.cuda.empty_cache()

        avg_grads = [g / total_samples for g in grads_sum]
        grads_np = [g.numpy() for g in avg_grads]

        del grads_sum, avg_grads
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return grads_np


    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # alpha = config["alpha"]
        self.log(
            f"{self.client_name} - Fit(): FedMeta client training started",
            object_id=self.id,
        )
        self.epoch_fl = config["server_round"]

        # === Load prototypes ===
        prots_file_path = f"prototypes_{self.client_name}.npz"
        self.criterion.load_state_dict(torch.load(prots_file_path, map_location=self.device))

        # === Load global model ===
        self.model.set_parameters(parameters)

        # === Clear any old adapted model ===
        if hasattr(self, "adapted_model"):
            del self.adapted_model
            torch.cuda.empty_cache()
            gc.collect()

        # === Use provided splits ===
        support_loader, query_loader = self.dataset.dataloader(validation=False)
        total_len = len(support_loader.dataset) + len(query_loader.dataset)

        # === Inner loop: adapt model on support (train set) ===
        self.adapted_model = copy.deepcopy(self.model).to(self.device)

        # Log prototype values BEFORE training this round
        with torch.no_grad():
            proto_slice = next(self.criterion.parameters()).view(-1)[:5].cpu().numpy()
            norm = sum(p.norm().item() for p in self.criterion.parameters())
            self.log(f"{self.client_name} - [START] Proto slice: {proto_slice}", object_id=self.id)
            self.log(f"{self.client_name} - [START] Proto norm: {norm:.4f}", object_id=self.id)

        if self.epoch_fl <= 5:
            loss = self.pre_train_model(self.adapted_model, support_loader, alpha=self.pre_train_lr, num_epochs=self.num_epochs)
        elif self.epoch_fl <= 40:
            loss = self.train_model(self.adapted_model, support_loader, alpha=self.lr, num_epochs=self.epochs, only_markers=True)
        else:
            loss = self.train_model(self.adapted_model, support_loader, alpha=self.lr, num_epochs=self.epochs, only_markers=False)

        with torch.no_grad():
            proto_slice = next(self.criterion.parameters()).view(-1)[:5].cpu().numpy()
            norm = sum(p.norm().item() for p in self.criterion.parameters())
            self.log(f"{self.client_name} - [END] Proto slice: {proto_slice}", object_id=self.id)
            self.log(f"{self.client_name} - [END] Proto norm: {norm:.4f}", object_id=self.id)

        # Save prototypes
        torch.save(self.criterion.state_dict(), prots_file_path)

        # === Outer loop: compute query loss on adapted model ===
        grads = self.compute_query_loss(self.adapted_model, query_loader)

        # Define your target file path
        grads_file_path = f"gradients_{self.client_name}.npz"
        np.savez(grads_file_path, *grads)

        # Free adapted model after training
        del self.adapted_model, grads
        torch.cuda.empty_cache()
        gc.collect()

        self.log(
            f"{self.client_name} - Fit() FedMeta client training finished",
            object_id=self.id,
        )

        return (
            self.model.get_parameters(),
            total_len,
            {"loss": loss, "grads_file": self.client_name},
        )

    ################################
    # TESTING  FUNCTIONS ######
    ################################

    def knn_in_embedding_space(self, model, train_loader, test_loader, k=1, num_samples_for_pairs=200):

        # collect train embeddings – use inference_mode and float16 on CPU
        train_emb_list, train_lbl_list = [], []
        with torch.inference_mode():
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device, non_blocking=True).float()
                emb = model(batch_data)
                train_emb_list.append(emb.detach().cpu().to(torch.float16))  # 2x smaller
                train_lbl_list.append(batch_labels.cpu())
                del emb, batch_data, batch_labels

        train_embeddings = torch.cat(train_emb_list).numpy()           # float16 numpy
        train_labels = torch.cat(train_lbl_list).numpy()

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings, train_labels)

        all_preds, all_labels = [], []
        running_loss, total_batches = 0.0, 0

        with torch.inference_mode():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device, non_blocking=True).float()
                emb = model(batch_data).detach().cpu().to(torch.float16).numpy()
                preds = knn.predict(emb)
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())

                # compute pair loss on embeddings (no extra forwards)
                # build pairs on GPU cheaply
                batch_labels_gpu = torch.from_numpy(batch_labels.numpy()).to(self.device)
                idx1, idx2, y = self.generate_pair_indices(batch_labels_gpu, num_samples_for_pairs)

                emb_gpu = torch.from_numpy(emb).to(self.device).to(torch.float32)  # cast back for criterion if needed
                e1 = emb_gpu.index_select(0, idx1)
                e2 = emb_gpu.index_select(0, idx2)
                loss = self.criterion(e1, e2, y)
                running_loss += float(loss.item())
                total_batches += 1

                del e1, e2, idx1, idx2, y, emb_gpu, emb, batch_data, batch_labels, batch_labels_gpu
                torch.cuda.empty_cache()

        self.log(
            f"{self.client_name} - Evaluate(): FedMeta Model evaluation started",
            object="experiment_evaluate",
            object_id=self.id,
        )

        avg_loss = running_loss / max(1, total_batches)
        y_pred  = torch.tensor(all_preds)
        y_label = torch.tensor(all_labels)
        return avg_loss, y_pred, y_label


    def test(self, model, trainloader, testloader):
        loss = 0.0
        model.to(self.device)
        model.eval()
        self.criterion.eval()

        accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
              average=None #"macro"
        ).to(self.device)

        precision = MulticlassPrecision(
            num_classes=self.num_classes, 
            average="macro"
        ).to(self.device)

        recall = MulticlassRecall(
            num_classes=self.num_classes, 
            average="macro"
        ).to(self.device)

        f1 = MulticlassF1Score(
            num_classes=self.num_classes, 
            average="macro"
            ).to(self.device)
        
        conf_matrix = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        ).to(self.device)

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

        acc = accuracy.compute().mean().item()          # macro average manually
        acc_per_class = accuracy.compute().cpu().numpy().tolist()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()
        conf = conf_matrix.compute().cpu().numpy()

        return float(loss), acc, acc_per_class, prec, rec, f1_score, conf

    def validation_loop(self, data_loaders, alpha):
        support_loader, query_loader = data_loaders

        # === Inner loop: adapt model on support (train set) ===
        self.adapted_model = copy.deepcopy(self.model).to(self.device)

        if self.epoch_fl <= 5:
            self.pre_train_model(self.adapted_model, support_loader, alpha=self.pre_train_lr, num_epochs=2)
        elif self.epoch_fl <= 40:
            self.train_model(self.adapted_model, support_loader, alpha=self.lr, num_epochs=2, only_markers=True)
        else:
            self.train_model(self.adapted_model, support_loader, alpha=self.lr, num_epochs=2, only_markers=False)

        # ---- Evaluate adapted model on query set ----
        loss_avg, acc, acc_per_class, prec, rec, f1, conf = self.test(
            self.adapted_model, support_loader, query_loader
        )

        # build per-class accuracy mapping
        acc_per_class_dict = {f"TEST_ACC_CLASS_{i}": v for i, v in enumerate(acc_per_class)}

        torch.cuda.empty_cache()

        return loss_avg, {
            "TEST_ACC": acc,
            "TEST_PREC": prec,
            "Recall": rec,
            "TEST_F1": f1,
            #**acc_per_class_dict  # add per-class metrics to the output dictionary
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        # alpha = config["alpha"]
        return_dic = {}

        self.log(
            f"{self.client_name} - Evaluate(): FedMeta Model evaluation started",
            object="experiment_evaluate",
            object_id=self.id,
        )

        self.model.set_parameters(parameters)

        self.epoch_fl = config["server_round"]
        loss, values_metrics_validation = self.validation_loop(
            self.dataset.dataloader(validation=True), self.lr
        )
        values_metrics_validation["LOSS"] = loss

        self.log(
            f"{self.client_name} - Measure: "
            + "metrics."
            + str(values_metrics_validation),
            details="",
            object="",
            object_id=self.id,
        )

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
