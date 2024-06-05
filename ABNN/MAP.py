"""

This file defines the key components of the Adaptive Bayesian Neural Network (ABNN) model, 
including the Bayesian Normalization Layer (BNL) and the custom Maximum A Posteriori (MAP) 
loss function. These components are essential in the implementation of the ABNN as described
in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from 
Pre-trained Models" (https://arxiv.org/abs/2312.15297).

"""

import torch
import torch.nn as nn

class ABNNLoss(torch.nn.Module):
    def __init__(self, num_classes=10, std=1.0, device=None):
        super(ABNNLoss, self).__init__()
        self.std = std
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta = nn.Parameter(torch.ones(num_classes, device=self.device))

    def forward(self, outputs, labels, model):
        # Calculate the three loss components
        nll_loss = self.negative_log_likelihood(outputs, labels)
        log_prior_loss = self.negative_log_prior(model, self.std)
        custom_ce_loss = self.custom_cross_entropy_loss(outputs, labels, self.eta)

        # Sum up all three components to form the ABNN loss
        total_loss = nll_loss + log_prior_loss + custom_ce_loss
        return total_loss

    @staticmethod
    def negative_log_likelihood(outputs, labels):
        # MLE Loss aka Negative Log Likelihood (NLL):
        # NLL = -∑ log P(y_i | x_i, ω)
        return torch.nn.functional.cross_entropy(outputs, labels)

    def negative_log_prior(self, model, std=1.0):
        # Negative Log Prior (L2 Regularization):
        # log P(ω) = - (1/2σ^2) * Σω_i^2 (Using Gaussian Prior) 
        # 1/2σ^2 acts as the weight decay
        std = torch.tensor(std, dtype=torch.float32)  
        variance = std ** 2
        weight_decay = torch.log(2 * torch.pi * variance) 
        total_log_prior = sum(
            0.5 * torch.sum((param / std) ** 2 + weight_decay)
            for param in model.parameters()
        )
        return total_log_prior

    def custom_cross_entropy_loss(self, outputs, labels, eta):
        # Custom Cross-Entropy Loss:
        # E(ω) = -∑ η_i log P(y_i | x_i, ω)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        weighted_log_probs = eta[labels] * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        return -torch.mean(weighted_log_probs)
