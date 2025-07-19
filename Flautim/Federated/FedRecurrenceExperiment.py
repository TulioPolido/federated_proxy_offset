from flautim.pytorch.federated.Experiment import Experiment
import flautim.metrics as flm
import torch
import numpy as np

class RecurrenceExperiment(Experiment):
    def __init__(self, model, dataset, context, lr=0.001, epochs=10, **kwargs):
        super(RecurrenceExperiment, self)\
            .__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
            )
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def balanced_accuracy(y_true, y_pred):
        # Flatten and convert to numpy array of scalars
        y_true = np.array([int(y.item() if hasattr(y, 'item') else y) for y in y_true])
        y_pred = np.array([int(y.item() if hasattr(y, 'item') else y) for y in y_pred])

        classes = np.unique(y_true)
        recalls = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

        return np.mean(recalls)

    flm.Metrics.balanced_accuracy = balanced_accuracy

    def training_loop(self, data_loader):
        self.model.train()
        loss_sum, all_preds, all_labels = 0.0, [], []


        for epoch in range(self.epochs):
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        balanced_acc = flm.Metrics.balanced_accuracy(all_labels, all_preds) * 100
        #precision = flm.Metrics.precision(all_labels, all_preds)

        return loss_sum / len(data_loader), {"ACC": balanced_acc}

    def validation_loop(self, data_loader):
        self.model.eval()
        loss_sum = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                loss_sum += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        balanced_acc = flm.Metrics.balanced_accuracy(all_labels, all_preds) * 100
        #precision = flm.Metrics.precision(all_labels, all_preds)

        return loss_sum / len(data_loader), {"ACC": balanced_acc}
