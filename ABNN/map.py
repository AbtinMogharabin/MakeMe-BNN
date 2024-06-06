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

    def forward(self, outputs, targets):
        device = outputs.device  # Get the device of the outputs tensor
        targets = targets.to(device)  # Move targets to the same device

        # Ensure eta is on the same device
        self.eta = self.eta.to(device)

        # Cross-entropy loss (negative log-likelihood)
        nll_loss = self.cross_entropy(outputs, targets)

        # Perturbation term: E(ω) = -∑ ηi logP(yi | xi, ω)
        perturbation_loss = self.eta[targets] * nll_loss

        # MAP loss: LMAP(ω) = -∑ logP(yi | xi, ω) - logP(ω)
        map_loss = nll_loss.mean() + self.prior_log_prob(self.model_parameters, self.prior_std)

        # Total loss: L(ω) = LMAP(ω) + E(ω)
        total_loss = map_loss + perturbation_loss.mean()

        return total_loss



    @staticmethod
    def prior_log_prob(params, std):
        # Compute logP(ω) for a normal prior with standard deviation `std`
        log_prob = 0.0
        for param in params:
            param = param.to(std)  # Move param to the same device as std
            log_prob += -0.5 * torch.sum(param ** 2) / (std ** 2)
        return log_prob



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
        log_prob = 0.0
        SS = sum((param ** 2).sum() for param in model.parameters()) # Sum of Squares
        negative_log_prior = 0.5 * SS / (std ** 2)
        return negative_log_prior

    def custom_cross_entropy_loss(self, outputs, labels, eta):
        # Custom Cross-Entropy Loss:
        # E(ω) = -∑ η_i log P(y_i | x_i, ω)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        weighted_log_probs = eta[labels] * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        return -torch.mean(weighted_log_probs)
