"""

This file defines the key components of the Adaptive Bayesian Neural Network (ABNN) model, 
including the Bayesian Normalization Layer (BNL) and the custom Maximum A Posteriori (MAP) 
loss function. These components are essential in the implementation of the ABNN as described
in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from 
Pre-trained Models" (https://arxiv.org/abs/2312.15297).

"""

import torch
import torch.nn as nn


class CustomMAPLoss(nn.Module):
    def __init__(self, eta, model_parameters, prior_std=1.0):
        super(CustomMAPLoss, self).__init__()
        self.eta = eta  # Class-dependent random weights
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.prior_std = prior_std
        self.model_parameters = model_parameters

    def forward(self, outputs, targets ):
        # Cross-entropy loss (negative log-likelihood)
        nll_loss = self.cross_entropy(outputs, targets)

        # Perturbation term: E(ω) = -∑ ηi logP(yi | xi, ω)
        perturbation_loss = self.eta[targets] * nll_loss

        # MAP loss: LMAP(ω) = -∑ logP(yi | xi, ω) - logP(ω)
        map_loss = nll_loss.mean() + self.prior_log_prob(self.model_parameters, prior_std)

        # Total loss: L(ω) = LMAP(ω) + E(ω)
        total_loss = map_loss + perturbation_loss.mean()

        return total_loss

    @staticmethod
    def prior_log_prob(params, std):
        # Compute logP(ω) for a normal prior with standard deviation `std`
        log_prob = 0.0
        for param in params:
            log_prob += -0.5 * torch.sum(param ** 2) / (std ** 2)
        return log_prob
