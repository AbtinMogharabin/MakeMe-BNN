import torch
import torch.nn as nn


class BNL(nn.Module):
    """
        Bayesian Normalization Layer (BNL).

    This layer replaces traditional normalization layers like BatchNorm,
    LayerNorm, and InstanceNorm. It adapts normalization to account for Bayesian
    inference, making the model more robust to variations and uncertainties in
    the data.

    BNL adds gaussian noise during both inference and trainig stages.

    This implementation includes parameters named `weight` and `bias` to directly
    match those used in PyTorch's BatchNorm, LayerNorm, and InstanceNorm layers 
    for compatibility when loading state dictionaries.

    Args:
        num_features (int, list, tuple): Number of features in the input, matches channels 
                                         in conv layers or features in linear layers. Can
                                         be a single integer or a list/tuple for complex scenarios.
    """
    def __init__(self, num_features):
        super(BNL, self).__init__()
        # Check if num_features is a list or tuple, convert if necessary
        if isinstance(num_features, int):
            num_features = (num_features,)
        
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x):
        if len(self.num_features) == 1:  # Traditional usage like BatchNorm
            mean = x.mean([0, 2, 3], keepdim=True) if x.dim() == 4 else x.mean(0, keepdim=True)
            var = x.var([0, 2, 3], keepdim=True) if x.dim() == 4 else x.var(0, keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            noise = torch.randn(self.weight.shape, device=x.device)
            gamma_noisy = self.weight * (1 + noise)

            if x.dim() == 4:
                gamma_noisy = gamma_noisy.view(1, -1, 1, 1)
                bias = self.bias.view(1, -1, 1, 1)
            elif x.dim() == 2:
                gamma_noisy = gamma_noisy.view(1, -1)
                bias = self.bias.view(1, -1)

            return gamma_noisy * x_normalized + bias
        else:  # LayerNorm-like usage
            mean = x.mean(dim=tuple(range(x.dim())[1:]), keepdim=True)
            var = x.var(dim=tuple(range(x.dim())[1:]), keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            noise = torch.randn(self.weight.shape, device=x.device)
            gamma_noisy = self.weight * (1 + noise)

            weight = self.weight.view((1,) + self.num_features + (1,) * (x.dim() - len(self.num_features) - 1))
            bias = self.bias.view((1,) + self.num_features + (1,) * (x.dim() - len(self.num_features) - 1))

            return gamma_noisy * x_normalized + bias


# class BNL(nn.Module):
#     def __init__(self, num_features, epsilon_std=1.0):
#         super(BNL, self).__init__()
#         self.num_features = num_features
#         self.epsilon_std = epsilon_std
        
#         # Learnable parameters
#         self.gamma = nn.Parameter(torch.ones(num_features))
#         self.beta = nn.Parameter(torch.zeros(num_features))
        
#         # Running statistics
#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)
        
#         # Initialize the random noise
#         self.epsilon = torch.randn(num_features) * self.epsilon_std

#     def forward(self, x):
#         device = x.device  # Get the device of the input tensor
        
#         # Ensure running statistics are on the same device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)
#         self.gamma = self.gamma.to(device)
#         self.beta = self.beta.to(device)
#         self.epsilon = self.epsilon.to(device)

#         if self.training:
#             batch_mean = torch.mean(x, dim=[0, 2, 3])
#             batch_var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            
#             # Update running statistics
#             self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
#             self.running_var = 0.9 * self.running_var + 0.1 * batch_var
#         else:
#             batch_mean = self.running_mean
#             batch_var = self.running_var

#         # Add Gaussian noise to gamma
#         if self.training:
#             self.epsilon = torch.randn(self.num_features, device=device) * self.epsilon_std
#         gamma_noisy = self.gamma * (1 + self.epsilon)

#         # Normalize the input
#         x_normalized = (x - batch_mean[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + 1e-5)
        
#         return gamma_noisy[None, :, None, None] * x_normalized + self.beta[None, :, None, None]
