import copy
import gc
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch
from flautim.pytorch.federated.Experiment import Experiment
from flwr.common.typing import NDArrays, Scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
from torch.cuda.amp import autocast


class RecurrenceExperiment(Experiment):
    def __init__(
        self,
        context,
        net,
        dataset,
        trainloaders,
        valloaders,
        cid=0,
        learning_rate=0.0001,
        num_epochs=1,
        gradient_step=2,
        num_classes=7,
        **kwargs,
    ):
        super(RecurrenceExperiment, self).__init__(net, dataset, context, **kwargs)
        self.net = net
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gradient_step = gradient_step
        self.num_classes = num_classes

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=0.001
        )

        self.net.to(self.device)
        self.adapted_model = copy.deepcopy(self.net).to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        model_keys = list(self.net.state_dict().keys())

        if len(model_keys) != len(parameters):
            self.log(f"{self.cid} ❌ PARAM COUNT MISMATCH", object_id=self.id)

        for i, (k, v) in enumerate(zip(model_keys, parameters)):
            if np.size(v) == 0:
                self.log("{self.cid}   ❌ EMPTY PARAMETER DETECTED", object_id=self.id)

        self.log(f"{self.cid} ----------------------------------", object_id=self.id)

        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    ################################
    # TRAINING FUNCTIONS ######
    ################################

    def train_model(self, model, train_loader, alpha, gradient_step=5):
        """Train with minimal memory footprint."""
        model.train()

        for _ in range(gradient_step): 
            total_loss, total_samples = (0.0, 0)  

            for (data, labels) in train_loader:  
                data, labels = (data.float().to(self.device), labels.long().to(self.device)) 

                # Mixed precision forward
                with autocast(enabled=True):  
                    outputs = model(data)  
                    loss = self.cross_entropy_loss(outputs, labels) 

                total_loss += loss.item() * labels.size(0)  
                total_samples += labels.size(0)  

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)  

                # Update parameters without tracking gradients
                with torch.no_grad(): 
                    for param, grad_ in zip(model.parameters(), grads):
                        if grad_ is not None:  
                            param -= (alpha * grad_) 

                # Clear tensors to avoid graph accumulation
                del (grads, outputs, data, labels, loss)  
                torch.cuda.empty_cache() 

            torch.cuda.synchronize()  

        return total_loss / total_samples 

    def compute_query_loss(self, model, query_loader):
        """Compute gradients for query set with minimal GPU memory usage."""
        #model.eval() # No Flower não existe esse passo. 
        total_samples = 0  
        loss_sum = 0.0

        for data, labels in query_loader:  
            data, labels = (data.float().to(self.device), labels.long().to(self.device))
            batch_size = labels.size(0)  
            total_samples += batch_size  

            # Use mixed precision for memory efficiency
            with torch.cuda.amp.autocast(enabled=True): 
                outputs = model(data)  
                loss = self.cross_entropy_loss(outputs, labels) 
                loss_sum += loss * labels.size(0)

            del (outputs, loss, data, labels)  
        
        loss_sum = loss_sum / total_samples
        grads = torch.autograd.grad(loss_sum, model.parameters(), create_graph=False) 

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        grads = [grad_.cpu().numpy() for grad_ in grads]

        torch.cuda.empty_cache() 
        torch.cuda.synchronize()  

        return grads, len(query_loader.dataset) 

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        torch.cuda.empty_cache()  
        torch.cuda.synchronize() 

        self.log(
            f"{self.cid} - FedMeta client training started",
            details="",
            object="",
            object_id=self.id,
        )
        self.epoch_fl = config["server_round"]

        # === Load global model ===
        self.set_parameters(parameters) 

        # === Clear any old adapted model ===
        if hasattr(self, "adapted_model"): 
            del self.adapted_model 
            torch.cuda.empty_cache() 
            gc.collect()  

        # === Reload dataloaders ===
        support_loader = self.trainloaders["sup"][self.cid]
        query_loader = self.trainloaders["qry"][self.cid]

        # === Create fresh model for adaptation ===
        self.adapted_model = copy.deepcopy(self.net).to(self.device) 

        # === Adapt model and calculate grads ===
        loss = self.train_model(
            self.adapted_model, 
            support_loader, 
            self.learning_rate, 
            self.gradient_step
        ) 

        grads, num_examples = self.compute_query_loss(self.adapted_model, query_loader) 

        file_path = f"gradients_{self.cid}.npz" 
        np.savez(file_path, *grads) 

        # === Clear memory ===
        del self.adapted_model, grads 
        torch.cuda.empty_cache() 
        gc.collect()  

        self.log(
            f"{self.cid} - FedMeta client finished",
            details="",
            object="",
            object_id=self.id,
        )

        return (
            self.get_parameters({}),
            num_examples,
            {"loss": loss, "grads_file": self.cid},
        )  

    ################################
    # TESTING FUNCTIONS ######
    ################################

    def validation_loop(self, alpha):
        # Necessario para o teste com o dataset FEMNIST pq o número de clientes de teste é muito menor.
        test_id = self.cid
        if test_id > len(self.valloaders["sup"]):
            test_id = self.cid % len(self.valloaders["sup"])

        support_loader = self.valloaders["sup"][test_id]
        query_loader = self.valloaders["qry"][test_id]

        # ---- Inner loop adaptation ----
        self.adapted_model = copy.deepcopy(self.net).to(self.device)
        self.train_model(self.adapted_model, support_loader, alpha, self.gradient_step)

        self.adapted_model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in query_loader:
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                outputs = self.adapted_model(X)
                total_loss += self.cross_entropy_loss(outputs, y).item()
                _, predicted = torch.max(outputs.detach(), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                del outputs, X, y
                torch.cuda.empty_cache()

        loss_avg = total_loss / len(query_loader)
        num_examples = len(query_loader.dataset)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted")

        torch.cuda.empty_cache()
        return loss_avg, num_examples, {
            "TEST_ACC": balanced_acc,
            "TEST_PREC": precision,
            "TEST_F1": f1,
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        return_dic = {}

        self.log(
            f"{self.cid} - FedMeta Model evaluation started",
            details="",
            object="experiment_evaluate",
            object_id=self.id,
        )

        self.set_parameters(parameters)
        self.epoch_fl = config["server_round"]

        loss, num_examples, values_metrics_validation = self.validation_loop(
            self.learning_rate
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
            f"{self.cid} - evaluation finished",
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
