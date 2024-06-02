"""
ABNN.py

This file defines the key components of the Adaptive Bayesian Neural Network (ABNN) model, 
including the Bayesian Normalization Layer (BNL) and the custom Maximum A Posteriori (MAP) 
loss function. These components are essential in the implementation of the ABNN as described
in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from 
Pre-trained Models" (https://arxiv.org/abs/2312.15297).

Classes:
    BNL: A custom Bayesian Normalization Layer that replaces traditional normalization 
         layers [Batch normalisation, Layer normalisation, and Instance normalisation]
         in deterministic neural networks to incorporate Bayesian inference.
    CustomMAPLoss: A custom loss function that combines the standard MAP loss with an 
                   additional epsilon term to manage Bayesian aspects.

Functions:
    replace_normalization_layers: A function to replace traditional normalization layers 
                                  in a given model's file with BNL and save the modified
                                  model with a new class name.
"""

import torch
import torch.nn as nn
import numpy as np
import re
import inspect

class BNL(nn.Module):
    """
    Bayesian Normalization Layer (BNL).

    This layer replaces traditional normalization including like BatchNorm, 
    LayerNorm, and InstanceNorm. It adapts normalization to account for Bayesian
    inference, making the model more robust to variations and uncertainties in 
    the data.

    Args:
        num_features (int): The number of features in the input.

    Methods:
        forward(x): Applies Bayesian normalization to the input tensor.
    """
    def __init__(self, num_features):
        super(BNL, self).__init__()
        self.num_features = num_features
        self.gamma_4d = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta_4d = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma_2d = nn.Parameter(torch.ones(1, num_features))
        self.beta_2d = nn.Parameter(torch.zeros(1, num_features))
        self.eps = 1e-5

    def forward(self, x):
        if x.dim() == 4:  # Input is 4D
            batch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            batch_var = torch.var(x, dim=(0, 2, 3), keepdim=True)
            noise = torch.randn(x.shape[0], self.num_features, 1, 1).to(x.device)
            x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            x = x * (self.gamma_4d * (1 + noise)) + self.beta_4d
        elif x.dim() == 2:  # Input is 2D
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True)
            noise = torch.randn(x.shape[0], self.num_features).to(x.device)
            x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            x = x * (self.gamma_2d * (1 + noise)) + self.beta_2d
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}")
        return x

class CustomMAPLoss(nn.Module):
    """
    Custom Maximum A Posteriori (MAP) Loss.

    This loss function combines the standard MAP loss with an additional epsilon term 
    to manage Bayesian aspects of the model. It incorporates class-dependent random 
    weights to account for uncertainty.

    Args:
        num_classes (int): Number of classes in the dataset.
        weight_decay (float): Weight decay factor for the prior term.

    Methods:
        forward(outputs, labels, model): Computes the total loss for the given outputs 
                                         and labels using the MAP and epsilon terms.
    """
    def __init__(self, num_classes, weight_decay=1e-4):
        super(CustomMAPLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.eta = np.random.uniform(0, 1, num_classes)  # Random weights for each class

    def forward(self, outputs, labels, model):
        # Compute Cross Entropy Loss
        ce_loss = self.criterion(outputs, labels)

        # Compute class-dependent random weights
        class_weights = torch.tensor([self.eta[label] for label in labels], device=outputs.device)
        weighted_loss = ce_loss * class_weights

        # Compute the log-likelihood loss
        log_likelihood_loss = ce_loss.mean()

        # Compute the prior term (weight decay)
        prior_loss = 0
        for param in model.parameters():
            prior_loss += torch.sum(param ** 2)
        prior_loss *= self.weight_decay / 2

        # Combine the MAP loss and the epsilon term
        map_loss = log_likelihood_loss + prior_loss
        epsilon_term = weighted_loss.mean()
        total_loss = map_loss + epsilon_term

        return total_loss

def replace_normalization_layers(original_model_path, new_model_name, save_new_model):
    """
    Replace normalization layers in a given model file with BNL.

    This function reads the model definition from the given file, replaces all instances 
    of traditional normalization layers (BatchNorm2d, BatchNorm1d, LayerNorm, InstanceNorm2d) 
    with BNL, renames the model class, and saves the modified model to a new file.

    Args:
        original_model_path (str): Path to the original model file.
        new_model_name (str): The new name for the model class.
        save_new_model (str): Location to save the new model.

    Methods:
        replace_normalization_layers(original_model_path, new_model_name, save_new_model): 
            Modifies the original model to use BNL and saves it with the new class name.
    """
    with open(original_model_path, 'r') as file:
        model_str = file.read()

    # Replace normalization layers with BNL
    norm_layers = [
        r'nn\.BatchNorm2d\(([^)]*)\)',
        r'nn\.BatchNorm1d\(([^)]*)\)',
        r'nn\.LayerNorm\(([^)]*)\)',
        r'nn\.InstanceNorm2d\(([^)]*)\)'
    ]
    for norm_layer in norm_layers:
        model_str = re.sub(norm_layer, r'BNL(\1)', model_str)

    # Extract the old class name
    old_class_name = re.search(r'class\s+(\w+)\(', model_str).group(1)

    # Rename the class to new_model_name
    model_str = re.sub(rf'class\s+{old_class_name}\(', f'class {new_model_name}(', model_str)
    
    # Rename all instances of the old class name in super() calls
    model_str = re.sub(rf'super\({old_class_name}', f'super({new_model_name}', model_str)

    # Add BNL import at the top
    model_str = 'from __main__ import BNL\n' + model_str

    # Save the modified model string to a new file
    with open(save_new_model, 'w') as file:
        file.write(model_str)
    print(f'Modified model saved to {save_new_model}')

# Usage Example:
# original_model_path = 'CNNNet.py' # The path of original DNN
# new_model_name = 'ABNNNet'        # The name of the new model
# save_new_model='ABNNNet.py'       # The file to save the new model
# replace_normalization_layers(original_model_path, new_model_name, save_new_model)
# Now export and check the new ABNN version of your original model
# from ABNNNet import ABNNNet
# model = ABNNNet()
# print(model)
