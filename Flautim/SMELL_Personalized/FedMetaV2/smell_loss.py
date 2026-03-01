# proxy_nca_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x1, x2, label, proxie_eff_sim=None, proxie_eff_dis=None):
        # Compute the absolute difference vector s_ij
        s_ij = torch.abs(x1 - x2)

        # Expand s_ij to compute distance to all prototypes
        s_ij = s_ij.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        # Compute distances to all similar prototypes
        if proxie_eff_sim is None:
            dist_to_similar_prototypes = torch.cdist(
                s_ij, self.similar_prototypes.unsqueeze(0), p=2
            ).squeeze(1)
        else:
            dist_to_similar_prototypes = torch.cdist(
                s_ij, proxie_eff_sim.unsqueeze(0), p=2
            ).squeeze(1)

        # Se os proxies dissimilares efetivos n forem nulos, eles sao utilizados no lugar dos proxies da loss
        if proxie_eff_dis is None:
            dist_to_dissimilar_prototypes = torch.cdist(
                s_ij, self.dissimilar_prototypes.unsqueeze(0), p=2
            ).squeeze(1)
        else:
            dist_to_dissimilar_prototypes = torch.cdist(
                s_ij, proxie_eff_dis.unsqueeze(0), p=2
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
