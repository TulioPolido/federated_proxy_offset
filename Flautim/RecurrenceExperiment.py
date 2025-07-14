from flautim.pytorch.federated.Experiment import Experiment
import flautim.metrics as flm
import torch


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


    # def accuracy_2(y, y_hat):
    #     y = np.asarray(y)
    #     y_hat = np.asarray(y_hat)
    #     return np.mean(y == y_hat)

    # flm.Metrics.accuracy_2 = accuracy_2

    def training_loop(self, data_loader):
        self.model.train()
        loss_sum, y_pred, y_true = 0.0, [], []

        for X, y in data_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted.cpu())
            y_true.append(y.cpu())

        accuracy = flm.Metrics.accuracy(
            torch.cat(y_pred).numpy(),
            torch.cat(y_true).numpy()
            )
        # accuracy_2 = flm.Metrics.accuracy_2(
        # torch.cat(y_pred).numpy(),
        # torch.cat(y_true).numpy()
        # )

        return loss_sum / len(data_loader), {"ACC": accuracy}

    def validation_loop(self, data_loader):
        self.model.eval()
        loss_sum, y_pred, y_true = 0.0, [], []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                loss_sum += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs, 1)
                y_pred.append(predicted.cpu())
                y_true.append(y.cpu())

        accuracy = flm.Metrics.accuracy(
            torch.cat(y_pred).numpy(),
            torch.cat(y_true).numpy()
            )
        # accuracy_2 = flm.Metrics.accuracy_2(
        # torch.cat(y_pred).numpy(),
        # torch.cat(y_true).numpy()
        # )

        return loss_sum / len(data_loader), {"ACC": accuracy}
