import torch
import torch.nn as nn

class BNL(nn.Module):
    def __init__(self, num_features, epsilon_std=1.0):
        super(BNL, self).__init__()
        self.num_features = num_features
        self.epsilon_std = epsilon_std
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        # Initialize the random noise
        self.epsilon = torch.randn(num_features) * self.epsilon_std

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        
        # Ensure running statistics are on the same device
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        self.epsilon = self.epsilon.to(device)

        if self.training:
            batch_mean = torch.mean(x, dim=[0, 2, 3])
            batch_var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
            self.running_var = 0.9 * self.running_var + 0.1 * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Add Gaussian noise to gamma
        if self.training:
            self.epsilon = torch.randn(self.num_features, device=device) * self.epsilon_std
        gamma_noisy = self.gamma * (1 + self.epsilon)

        # Normalize the input
        x_normalized = (x - batch_mean[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + 1e-5)
        
        return gamma_noisy[None, :, None, None] * x_normalized + self.beta[None, :, None, None]

