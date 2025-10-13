from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np
import gc

from flautim.pytorch.federated.Experiment import Experiment
import torch
import random
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from flwr.common.typing import NDArrays, Scalar
import copy

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    f1_score,
)

from torch.cuda.amp import autocast


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
        **kwargs,
    ):
        super(RecurrenceExperiment, self).__init__(model, dataset, context, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.epochs = epochs
        self.num_classes = num_classes
        self.lr = lr
        self.pre_train_lr = pre_train_lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.client_name = client_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.adapted_model = copy.deepcopy(self.model).to(self.device)

        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)

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

    def train_model(self, model, train_loader, alpha, num_epochs=1):
        """Train with minimal memory footprint."""
        for _ in range(num_epochs):
            total_loss, total_samples = 0.0, 0

            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Mixed precision forward
                with autocast(enabled=True):
                    outputs = model(data)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)

                # Update parameters without tracking gradients
                with torch.no_grad():
                    for param, grad_ in zip(model.parameters(), grads):
                        if grad_ is not None:
                            param -= alpha * grad_

                # Clear tensors to avoid graph accumulation
                del grads, outputs, data, labels, loss
                torch.cuda.empty_cache()

            torch.cuda.synchronize()

        return total_loss / total_samples


    def compute_query_loss(self, model, query_loader):
        """Compute gradients for query set with minimal GPU memory usage."""
        model.eval()
        grads_sum = [torch.zeros_like(p, device="cpu") for p in model.parameters()]
        total_samples = 0

        for data, labels in query_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # Use mixed precision for memory efficiency
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(data)
                loss = self.criterion(outputs, labels)

            # Compute per-batch grads, without graph
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)

            # Accumulate grads safely on CPU
            for i, g in enumerate(grads):
                if g is not None:
                    grads_sum[i] += g.detach().cpu() * batch_size

            del outputs, loss, grads, data, labels
            torch.cuda.empty_cache()

        # Average grads across all samples
        avg_grads = [g / total_samples for g in grads_sum]

        # Convert to numpy for Flower
        grads_np = [g.numpy() for g in avg_grads]

        del grads_sum, avg_grads
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return grads_np


    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            self.log(f"{self.client_name} - FedMeta client training started", details="", object="", object_id=self.id)
            self.epoch_fl = config["server_round"]

            # === Load global model ===
            self.model.set_parameters(parameters)

            # === Clear any old adapted model ===
            if hasattr(self, "adapted_model"):
                del self.adapted_model
                torch.cuda.empty_cache()
                gc.collect()

            # === Reload dataloaders ===
            support_loader, query_loader = self.dataset.dataloader(validation=False)
            total_len = len(support_loader.dataset) + len(query_loader.dataset)

            # === Create fresh model for adaptation ===
            self.adapted_model = copy.deepcopy(self.model).to(self.device)

            loss = self.train_model(self.adapted_model, support_loader, self.lr, self.epochs)

            grads = self.compute_query_loss(self.adapted_model, query_loader)

            file_path = f"gradients_{self.client_name}.npz"
            np.savez(file_path, *grads)

            # Free adapted model after training
            del self.adapted_model, grads
            torch.cuda.empty_cache()
            gc.collect()

            self.log(
                f"{self.client_name} - FedMeta client finished",
                details="",
                object="",
                object_id=self.id,
            )

            return (
                self.model.get_parameters(),
                total_len,
                {"loss": loss, "grads_file": self.client_name},
            )


    ################################
    # TESTING FUNCTIONS ######
    ################################

    def validation_loop(self, data_loaders, alpha):
        support_loader, query_loader = data_loaders

        # ---- Inner loop adaptation ----
        self.adapted_model = copy.deepcopy(self.model).to(self.device)
        loss = self.train_model(
            self.adapted_model, support_loader, alpha, 2
        )

        self.adapted_model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in query_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.adapted_model(X)
                total_loss += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs.detach(), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                del outputs, X, y
                torch.cuda.empty_cache()

        loss_avg = total_loss / len(query_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted")

        torch.cuda.empty_cache()
        return loss_avg, {
            "TEST_ACC": balanced_acc,
            "TEST_PREC": precision,
            "TEST_F1": f1,
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        return_dic = {}
        try:
            self.log(
                f"{self.client_name} - FedMeta Model evaluation started",
                details="",
                object="experiment_evaluate",
                object_id=self.id,
            )

            self.model.set_parameters(parameters)
            self.epoch_fl = config["server_round"]

            loss, values_metrics_validation = self.validation_loop(
                self.dataset.dataloader(validation=True), self.lr
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
        except Exception as e:
            self.log(
                f"{self.client_name} - ERROR in evaluate(): {e}",
                details="",
                object="experiment_evaluate",
                object_id=self.id,
            )
            raise
