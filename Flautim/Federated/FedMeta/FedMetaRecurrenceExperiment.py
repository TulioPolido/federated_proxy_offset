from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np

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
    # confusion_matrix,
)

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
        
        # self.pre_train = pre_train
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
        for _ in range(num_epochs):
            loss_sum = 0.0
            sup_num_sample = []
            sup_total_loss = []

            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                loss_sum += loss.item()
                sup_num_sample.append(labels.size(0))
                sup_total_loss.append(loss * labels.size(0))

                grads = torch.autograd.grad(
                    loss, list(model.parameters()), create_graph=False
                )

                for param, grad_ in zip(model.parameters(), grads):
                    param.data = param.data - alpha * grad_

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        loss = sum(sup_total_loss) / sum(sup_num_sample)
        return loss

    def compute_query_loss(self, model, query_loader):
        model.eval()  # just evaluation mode, but gradients are tracked
        qry_total_loss = []
        qry_num_sample = []
        loss_sum = 0.0

        for data, labels in query_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss_sum += loss.item()

            qry_num_sample.append(labels.size(0))
            qry_total_loss.append(loss.item())

        loss_sum = loss_sum / sum(qry_num_sample)
        grads = torch.autograd.grad(loss_sum, list(model.parameters()))

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        grads = [grad_.cpu().numpy() for grad_ in grads]
        return grads

    def fit(  # ignorar
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        try:
            # alpha = config["alpha"]
            self.log(
                f"{self.client_name} - FedMeta client training started",
                details="",
                object="",
                object_id=self.id,
            )
            self.epoch_fl = config["server_round"]

            # === Load global model ===
            self.model.set_parameters(parameters)

            # === Use provided splits ===
            support_loader, query_loader = self.dataset.dataloader(validation=False)
            total_len = len(support_loader.dataset) + len(query_loader.dataset)

            # === Inner loop: adapt model on support (train set) ===
            self.adapted_model = copy.deepcopy(self.model).to(self.device)

            loss = self.train_model(
                self.adapted_model, support_loader, self.lr, self.epochs
            )

            # === Outer loop: compute query loss on adapted model ===
            grads = self.compute_query_loss(self.adapted_model, query_loader)  # ignorar
            
            # Define your target file path
            file_path = f"gradients_{self.client_name}.npz"
            np.savez(file_path, *grads)

            self.log(
                f"{self.client_name} - FedMeta client finished",
                details="",
                object="",
                object_id=self.id,
            )

            # grads is already a list of numpy arrays
            return (
                self.model.get_parameters(),
                total_len,
                {"loss": loss.item(), "grads_file": self.client_name},
            )
        except Exception as e:
            self.log(
                f"{self.client_name} - ERROR in fit(): {e}",
                details="",
                object="experiment_fit",
                object_id=self.id,
            )
            raise

    ################################
    # TESTING  FUNCTIONS ######
    ################################


    def validation_loop(self, data_loaders, alpha):
        support_loader, query_loader = data_loaders

        # ---- Inner loop adaptation ----
        self.adapted_model = copy.deepcopy(self.model).to(self.device)
        loss = self.train_model(
            self.adapted_model, support_loader, alpha, self.epochs
        )

        # ---- Evaluate adapted model on query set ----
        loss_avg, acc, prec, rec, f1, conf = self.test(
            self.adapted_model, support_loader, query_loader
        )

        self.adapted_model.eval()
        loss_sum = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in query_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.adapted_model(X)
                loss_sum += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs.detach(), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        loss_avg = loss_sum / len(query_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return loss_avg, {
            "TEST_ACC": balanced_acc,
            "TEST_PREC": precision,
            "TEST_F1": f1,
        }

    def evaluate(  # ignorar
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        # alpha = config["alpha"]
        return_dic = {}

        # self.log_parameters(parameters, "evaluate")

        try:
            self.log(
                f"{self.client_name} - FedMeta Model evaluation started",
                details="",
                object="experiment_evaluate",
                object_id=self.id,
            )

            self.model.set_parameters(parameters)

            self.epoch_fl = config["server_round"]
            self.log(
                f"{self.client_name} - Val Loop",
                details="",
                object="experiment_evaluate",
                object_id=self.id,
            )
            loss, values_metrics_validation = self.validation_loop(
                self.dataset.dataloader(validation=True), self.lr
            )
            values_metrics_validation["LOSS"] = loss

            self.log(
                f"{self.client_name} - Mesure: "
                + "metrics."
                + str(values_metrics_validation),
                details="",
                object="",
                object_id=self.id,
            )

            for name in values_metrics_validation:
                # self.log(f"Mesure: "+ 'metrics.' + str(name) , details="", object="", object_id=self.id)
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
