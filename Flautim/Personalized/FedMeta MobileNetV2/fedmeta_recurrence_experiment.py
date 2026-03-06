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
from torch.optim.lr_scheduler import StepLR


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
        label_smoothing=0.1,
        **kwargs,
    ):
        super(RecurrenceExperiment, self).__init__(model, dataset, context, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
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
        for _ in range(num_epochs):  # Para cada epoca
            total_loss, total_samples = (
                0.0,
                0,
            )  # Inicializa a loss total e samples total

            for (
                data,
                labels,
            ) in train_loader:  # Para cada par de dados e labels no trainloader
                data, labels = (
                    data.to(self.device),
                    labels.to(self.device),
                )  # Passa os dados e labels para a GPU

                # Mixed precision forward
                with autocast(
                    enabled=True
                ):  # Autocast seleciona de forma dinamica e automatica a precisao numerica
                    outputs = model(
                        data
                    )  # Passa os dados pelo modelo e guarda as saidas geradas
                    loss = self.criterion(
                        outputs, labels
                    )  # Utiliza entropia cruzada para calcular a loss

                total_loss += loss.item() * labels.size(
                    0
                )  # Total Loss é somado à loss atual * a quantidade de labels
                total_samples += labels.size(
                    0
                )  # Samples total é somado à quantidade de labels

                grads = torch.autograd.grad(
                    loss, list(model.parameters()), create_graph=False
                )  # Calcula o gradiente da loss em relação aos parametros do modelo

                # Update parameters without tracking gradients
                with torch.no_grad():  # Ignora os gradientes no bloco
                    for param, grad_ in zip(
                        model.parameters(), grads
                    ):  # Para cada par parametro-gradiente
                        if grad_ is not None:  # Se o gradiente é não nulo
                            param -= (
                                alpha * grad_
                            )  # Subtrai-se do parametro o gradiente respectivo * uma constante alpha

                # Clear tensors to avoid graph accumulation
                del (
                    grads,
                    outputs,
                    data,
                    labels,
                    loss,
                )  # deleta variáveis para economizar espaço
                torch.cuda.empty_cache()  # Esvazia a cache da GPU

            torch.cuda.synchronize()  # forca a CPU a aguardar as operacoes CUDA ao fim da época

        return total_loss / total_samples  # Retorna o valor de loss média

    def compute_query_loss(self, model, query_loader):
        """Compute gradients for query set with minimal GPU memory usage."""
        model.eval()  # Altera o modo do modelo para avaliação
        grads_sum = [
            torch.zeros_like(p, device="cpu") for p in model.parameters()
        ]  # Inicializa o vetor de somatorio de gradientes no formato do vetores de parametros
        total_samples = 0  # Inicializa samples totais

        for data, labels in query_loader:  # Para cada par dado-label no quaryloader
            data, labels = (
                data.to(self.device),
                labels.to(self.device),
            )  # Carrega os dados e labels na GPU
            batch_size = labels.size(0)  # Inicializa o tamanho do batch
            total_samples += batch_size  # Soma o tamanho do batch ao total de samples

            # Use mixed precision for memory efficiency
            with torch.cuda.amp.autocast(
                enabled=True
            ):  # Aplica precisao numérica automática
                outputs = model(data)  # Envia os dados para o modelo e salva a saída
                loss = self.criterion(
                    outputs, labels
                )  # Calcula a perda com entropia cruzada

            # Compute per-batch grads, without graph
            grads = torch.autograd.grad(
                loss, list(model.parameters()), create_graph=False
            )  # Computa os gradientes da loss em relação aos parametros do modelo

            # Accumulate grads safely on CPU
            for i, g in enumerate(
                grads
            ):  # Para cada par índice-gradiente na lista de gradientes
                if g is not None:  # Se o gradiente não é nulo
                    grads_sum[i] += (
                        g.detach().cpu() * batch_size
                    )  # Soma na posição do índice em grads_sum o gradiente multiplicado pelo batch_size

            del (
                outputs,
                loss,
                grads,
                data,
                labels,
            )  # Apaga variáveis para economizar memória
            torch.cuda.empty_cache()  # Limpa cache da GPU

        # Average grads across all samples
        avg_grads = [
            g / total_samples for g in grads_sum
        ]  # Calcula o gradiente médio em cada posição da lista grads_sum

        # Convert to numpy for Flower
        grads_np = [
            g.numpy() for g in avg_grads
        ]  # Converte os gradiente médios para numpy

        del grads_sum, avg_grads  # Apaga variáveis para economizar memória
        torch.cuda.empty_cache()  # Limpa cache da GPU
        torch.cuda.synchronize()  # forca a CPU a aguardar as operacoes CUDA

        return grads_np  # retorna os gradientes como numpy

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        torch.cuda.empty_cache()  # Limpa cache da GPU
        torch.cuda.synchronize()  # Força CPU a aguardar operações CUDA

        self.log(
            f"{self.client_name} - FedMeta client training started",
            details="",
            object="",
            object_id=self.id,
        )
        self.epoch_fl = config["server_round"]

        # === Load global model ===
        self.model.set_parameters(parameters)  # Salva os parâmetros recebidos no modelo

        # === Clear any old adapted model ===
        if hasattr(self, "adapted_model"):  # Se existir modelo adaptado
            del self.adapted_model  # Apaga o modelo adaptado
            torch.cuda.empty_cache()  # Limpa a cache da GPU
            gc.collect()  # Força coleta de lixo

        # === Reload dataloaders ===
        support_loader, query_loader = self.dataset.dataloader(
            validation=False
        )  # Recebe os dados de treino e teste
        total_len = len(support_loader.dataset) + len(
            query_loader.dataset
        )  # Salva o tamanho total dos dados somados

        # === Create fresh model for adaptation ===
        self.adapted_model = copy.deepcopy(self.model).to(
            self.device
        )  # Cria um modelo adaptado (uma cópia do modelo atual)

        loss = self.train_model(
            self.adapted_model, support_loader, self.lr, self.epochs
        )  # Treina o modelo adaptado com dados de treinamento

        grads = self.compute_query_loss(
            self.adapted_model, query_loader
        )  # Calcula o gradiente do modelo adaptado com os dados de teste

        file_path = f"gradients_{self.client_name}.npz"  # Cria um nome para o arquivo de gradientes
        np.savez(file_path, *grads)  # Salva os gradientes desse cliente no arquivo

        # Free adapted model after training
        del self.adapted_model, grads  # Deleta o modelo adaptado e os gradientes
        torch.cuda.empty_cache()  # Limpa a cache da GPU
        gc.collect()  # Força coleta de lixo

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
        )  # Retorna os parâmetros originais do modelo, o tamanho total dos dados, a loss e o nome do arquivo de gradientes

    ################################
    # TESTING FUNCTIONS ######
    ################################

    def validation_loop(self, data_loaders, alpha):
        support_loader, query_loader = data_loaders

        # ---- Inner loop adaptation ----
        self.adapted_model = copy.deepcopy(self.model).to(self.device)
        self.train_model(self.adapted_model, support_loader, alpha, 2)

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
