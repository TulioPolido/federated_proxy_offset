from flautim.pytorch.federated.Experiment import Experiment
import flautim.metrics as flm
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import balanced_accuracy_score
import flautim as fl

class RecurrenceExperiment(Experiment):
    def __init__(self, model, dataset, context, lr=0.001, epochs=10, weight_decay=1e-4, lr_step=3, lr_gamma=0.5, **kwargs):
        super(RecurrenceExperiment, self)\
            .__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
            )
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def training_loop(self, data_loader):
        try:
            self.model.train()
            loss_sum = 0.0
            all_preds, all_labels = [], []

            for data, labels in data_loader:
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

            loss_avg = loss_sum / len(data_loader)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            #balanced_acc = self.metrics.ACCURACY(all_labels, all_preds) * 100
            #precision = flm.Metrics.precision(all_labels, all_preds)

            self.scheduler.step()
            return loss_avg, {"TRAIN_ACC": balanced_acc}
        except Exception as e:
            fl.log(f"Error in training loop: {e}")

    def validation_loop(self, data_loader):
        try:
            self.model.eval()
            loss_sum = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for X, y in data_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    outputs = self.model(X)
                    loss_sum += self.criterion(outputs, y).item()
                    _, predicted = torch.max(outputs.detach(), 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            loss_avg = loss_sum / len(data_loader)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            #balanced_acc = self.metrics.ACCURACY(all_labels, all_preds) * 100
            #precision = flm.Metrics.precision(all_labels, all_preds)

            return loss_avg, {"TEST_ACC": balanced_acc}
        except Exception as e:
            fl.log(f"Error in validation loop: {e}")
