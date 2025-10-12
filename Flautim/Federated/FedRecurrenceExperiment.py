from flautim.pytorch.federated.Experiment import Experiment
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    f1_score,
    # confusion_matrix,
)
import flautim as fl


class RecurrenceExperiment(Experiment):
    def __init__(
        self,
        model,
        dataset,
        context,
        client_name=0,
        lr=0.001,
        epochs=10,
        weight_decay=1e-4,
        lr_step=3,
        lr_gamma=0.5,
        **kwargs,
    ):
        try:
            super(RecurrenceExperiment, self).__init__(
                model, dataset, context, **kwargs
            )
            self.client_name = client_name
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)
            self.epochs = epochs
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            fl.log(f"Error in RecurrenceExperiment __init__: {e}")
            raise

    def training_loop(self, train_loader):
        try:
            self.model.train()
            loss_sum = 0.0
            all_preds, all_labels = [], []

            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                _, predicted = torch.max(outputs.detach(), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            loss_avg = loss_sum / len(train_loader)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            precision = precision_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
            f1 = f1_score(all_labels, all_preds, average="weighted")

            self.scheduler.step()
            return loss_avg, {
                "TRAIN_ACC": balanced_acc,
                "TRAIN_PREC": precision,
                "TRAIN_F1": f1,
            }
        except Exception as e:
            fl.log(f"Error in training loop: {e}")
            raise

    def validation_loop(self, test_loader):
        try:
            self.model.eval()
            loss_sum = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(X)
                    loss_sum += self.criterion(outputs, y).item()
                    _, predicted = torch.max(outputs.detach(), 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            loss_avg = loss_sum / len(test_loader)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            precision = precision_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
            f1 = f1_score(all_labels, all_preds, average="weighted")
            # conf_matrix = confusion_matrix(all_labels, all_preds, normalize="true")

            return loss_avg, {
                "TEST_ACC": balanced_acc,
                "TEST_PREC": precision,
                "TEST_F1": f1,
                # "TEST_CONF": str(conf_matrix),
            }
        except Exception as e:
            fl.log(f"Error in validation loop: {e}")
            raise
