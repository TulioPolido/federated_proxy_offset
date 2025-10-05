from flautim.pytorch.federated.Experiment import Experiment
import torch
import torch.nn.functional as F
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
        client_name=None,
        lr=0.001,
        pre_train_lr=0.001,
        epochs=10,
        weight_decay=1e-4,
        lr_step=3,
        lr_gamma=0.5,
        num_classes=7,
        embedding_size=64,
        margin=1.0,
        **kwargs,
    ):

        super(RecurrenceExperiment, self).__init__(model, dataset, context, **kwargs)

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

        self.criterion = LearnablePrototypicalLossWithDiversityRegularization(
            embedding_dim=self.embedding_size,
            num_similar_prototypes=2,
            num_dissimilar_prototypes=2,
        ).to(self.device)
        self.optimizer1 = torch.optim.Adam(
            list(self.criterion.parameters()), lr=self.lr
        )
        self.optimizer2 = torch.torch.optim.SGD(
            [{"params": self.model.parameters(), "lr": self.lr}]
            + [{"params": self.criterion.parameters(), "lr": self.lr}]
        )
        self.pre_criterion = PrototypicalCenterLoss(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            margin=self.margin,
        ).to(self.device)
        self.pre_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.pre_criterion.parameters()),
            lr=self.pre_train_lr,
        )

        self.scheduler = StepLR(self.optimizer1, step_size=lr_step, gamma=lr_gamma)

    ################################
    # TRAINING FUNCTIONS ######
    ################################

    # Modify the pretrain function to use prototypical center loss
    def pretrain_model_prototypical_center(self, dataloader, num_epochs):
        self.model.train()  # Set the self.model to training mode

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_data, batch_labels in dataloader:
                # batch_data = batch_data.view(batch_data.size(0), -1).to(device)
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass: compute embeddings
                embeddings = self.model(batch_data.float())

                # Compute prototypical center loss
                loss = self.pre_criterion(embeddings, batch_labels)

                # Backpropagation and optimization
                self.pre_optimizer.zero_grad()
                loss.backward()
                self.pre_optimizer.step()

                running_loss += loss.item()

            # fl.log(f'Pretrain Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    def generate_pairs(self, X, Y, N):
        # Ensure N is even
        if N % 2 != 0:
            raise ValueError("N must be an even number.")

        num_same_label_pairs = N // 2
        num_different_label_pairs = N // 2

        # Precompute label indices
        label_to_indices = {
            label.item(): (Y == label).nonzero(as_tuple=True)[0].tolist()
            for label in torch.unique(Y)
        }

        X1, X2, labels = [], [], []

        # Generate similar pairs (same label)
        for _ in range(num_same_label_pairs):
            # Sample a label that has more than one instance
            while True:
                sampled_label = random.choice(list(label_to_indices.keys()))
                indices = label_to_indices[sampled_label]
                if len(indices) > 1:
                    # Randomly sample two distinct indices for the same label
                    idx1, idx2 = random.sample(indices, 2)
                    X1.append(X[idx1])
                    X2.append(X[idx2])
                    labels.append(1)  # Similar pair label
                    # Ensure the label has more than one element
                    break

        # Generate dissimilar pairs (different labels)
        for _ in range(num_different_label_pairs):
            # Sample two different labels
            while len(list(label_to_indices.keys())) >= 2:

                label1, label2 = random.sample(list(label_to_indices.keys()), 2)
                idx1 = random.choice(label_to_indices[label1])
                idx2 = random.choice(label_to_indices[label2])
                # Directly add the dissimilar pairs
                X1.append(X[idx1])
                X2.append(X[idx2])
                labels.append(0)  # Dissimilar pair label
                break

        # Convert lists to tensors efficiently in one go
        X1 = torch.stack(X1)
        X2 = torch.stack(X2)
        labels = torch.tensor(labels)

        return X1, X2, labels

    # Training loop with dynamic learning rate adjustment
    def train_model_with_dynamic_lr(
        self, train_loader, optimizer, num_epochs, frozen=False
    ):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=self.lr_gamma
        )  # Reduce LR by 0.5 every 50 epochs

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                # Create pairs of samples and labels (1 for similar, 0 for dissimilar)
                # pairs, labels = create_pairs(batch_data, batch_labels)
                if True:
                    pairs_1, pairs_2, labels = self.generate_pairs(
                        batch_data, batch_labels, 200
                    )
                    pairs_1, pairs_2, labels = (
                        pairs_1.to(self.device),
                        pairs_2.to(self.device),
                        labels.to(self.device),
                    )

                # Compute embeddings using the frozen encoder
                if frozen:
                    with torch.no_grad():
                        # Compute embeddings for both elements of the pairs
                        embeddings_x1 = self.model(pairs_1.float())
                        embeddings_x2 = self.model(pairs_2.float())
                        self.criterion.toggle_requires_grad(requires_grad_value=False)

                else:
                    # Compute embeddings for both elements of the pairs
                    embeddings_x1 = self.model(pairs_1.float())
                    embeddings_x2 = self.model(pairs_2.float())
                    self.criterion.toggle_requires_grad(requires_grad_value=True)

                # Compute loss
                loss = self.criterion(embeddings_x1, embeddings_x2, labels)

                # Backpropagation and optimization

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            return running_loss / len(train_loader)

    def training_loop(self, data_loaders):
        train_loader, _ = data_loaders
        self.pretrain_model_prototypical_center(train_loader, 2)
        loss_avg1 = self.train_model_with_dynamic_lr(
            train_loader, self.optimizer1, self.epochs
        )
        loss_avg2 = self.train_model_with_dynamic_lr(
            train_loader, self.optimizer2, self.epochs
        )
        return (loss_avg1 + loss_avg2) / 2, {}

    def fit(self, parameters, config):
        return_dic = {}

        self.log(
            f"{self.client_name} - Model training started",
            details="",
            object="",
            object_id=self.id,
        )

        self.model.set_parameters(parameters)

        self.epoch_fl = config["server_round"]

        for epochs in range(1, self.epochs + 1):
            loss, values_metrics_train = self.training_loop(self.dataset.dataloader())
            values_metrics_train["LOSS"] = loss
        for name in values_metrics_train:
            self.measures.log(
                self,
                name,
                values_metrics_train[name],
                validation=False,
                epoch=self.epoch_fl,
            )
            return_dic[name] = float(values_metrics_train[name])

        self.log(
            f"{self.client_name} - Model training finished",
            details="",
            object="",
            object_id=self.id,
        )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), return_dic

    ################################
    # TESTING  FUNCTIONS ######
    ################################

    def knn_in_embedding_space(
        self, train_loader, test_loader, k=1, num_samples_for_pairs=20
    ):
        self.model.eval()

        # Collect train embeddings and labels
        train_embeddings = []
        train_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in train_loader:
                # batch_data = batch_data.view(batch_data.size(0), -1).to(self.device)
                batch_data = batch_data.to(self.device)
                embeddings = self.model(batch_data)
                train_embeddings.append(embeddings.cpu())
                train_labels.append(batch_labels)

        train_embeddings = torch.cat(train_embeddings).cpu().numpy()
        train_labels = torch.cat(train_labels).cpu().numpy()

        # Initialize the k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings, train_labels)

        # Store predictions, true labels, and loss values
        all_preds = []
        all_labels = []
        running_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                # batch_data = batch_data.view(batch_data.size(0), -1).to(self.device)
                batch_data = batch_data.to(self.device)
                embeddings = self.model(batch_data).cpu().numpy()

                # Perform k-NN classification in embedding space
                predicted_labels = knn.predict(embeddings)

                # Store predictions and true labels
                all_preds.extend(predicted_labels)
                all_labels.extend(batch_labels.cpu().numpy())

                # Compute loss based on pairs in the test set

                # Generate pairs for loss computation
                pairs_1, pairs_2, labels = self.generate_pairs(
                    batch_data, batch_labels, num_samples_for_pairs
                )
                pairs_1, pairs_2, labels = (
                    pairs_1.to(self.device),
                    pairs_2.to(self.device),
                    labels.to(self.device),
                )

                # Compute embeddings for pairs
                embeddings_x1 = self.model(pairs_1)
                embeddings_x2 = self.model(pairs_2)

                # Compute loss
                loss = self.criterion(embeddings_x1, embeddings_x2, labels)
                running_loss += loss
                total_batches += 1

        # Average loss over all batches
        if total_batches > 0:
            avg_loss = running_loss / total_batches
        else:
            avg_loss = 0.0

        print(f"Average loss on test set: {avg_loss:.4f}")

        # Convert predictions and true labels to tensors for consistency
        y_pred = torch.tensor(all_preds)
        y_label = torch.tensor(all_labels)

        return avg_loss, y_pred, y_label

    def test(self, trainloader, testloader):
        loss = 0.0
        self.model.to(self.device)
        self.model.eval()

        accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="macro").to(
            self.device
        )
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

        loss, predicted, label = self.knn_in_embedding_space(trainloader, testloader)
        predicted = torch.tensor(predicted).to(self.device)
        label = torch.tensor(label).to(self.device)

        accuracy.update(predicted, label)
        precision.update(predicted, label)
        recall.update(predicted, label)
        f1.update(predicted, label)
        conf_matrix.update(predicted, label)

        acc = accuracy.compute().item()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()
        conf = conf_matrix.compute().cpu().numpy()

        return float(loss), acc, prec, rec, f1_score, conf

    def validation_loop(self, data_loaders):
        train_loader, test_loader = data_loaders
        self.pretrain_model_prototypical_center(train_loader, 1)
        loss_avg1 = self.train_model_with_dynamic_lr(train_loader, self.optimizer1, 1)
        loss_avg2 = self.train_model_with_dynamic_lr(train_loader, self.optimizer2, 1)
        loss_avg, acc, prec, rec, f1, conf_matrix = self.test(train_loader, test_loader)
        return loss_avg, {
            "TEST_ACC": acc,
            "TEST_PREC": prec,
            "Recall": rec,
            "TEST_F1": f1,
            # "TEST_CONF": str(conf_matrix),
        }

    def evaluate(self, parameters, config):

        return_dic = {}

        self.log(
            f"{self.client_name} - Model evaluation started",
            details="",
            object="experiment_evaluate",
            object_id=self.id,
        )

        self.model.set_parameters(parameters)

        self.epoch_fl = config["server_round"]

        loss, values_metrics_validation = self.validation_loop(
            self.dataset.dataloader(validation=True)
        )
        values_metrics_validation["LOSS"] = loss

        self.log(
            f"{self.client_name} - Model training finished",
            details="",
            object="experiment_evaluate",
        )

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

        return float(loss), len(self.dataset.dataloader(validation=True)), return_dic
