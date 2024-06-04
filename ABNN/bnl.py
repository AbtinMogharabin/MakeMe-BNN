import torch
import torch.nn as nn

class BNL(nn.Module):
    def __init__(self, num_features, epsilon_std=1.0):
        super(BNL, self).__init__()
        self.num_features = num_features
        self.epsilon_std = epsilon_std
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.epsilon = torch.randn(num_features) * self.epsilon_std

    def forward(self, x):
        if self.training:
            # Compute batch mean and variance over the appropriate dimensions
            if x.dim() == 2:  # If input is (batch_size, num_features)
                batch_mean = torch.mean(x, dim=0)
                batch_var = torch.var(x, dim=0, unbiased=False)
            elif x.dim() == 4:  # If input is (batch_size, num_channels, height, width)
                batch_mean = torch.mean(x, dim=[0, 2, 3])
                batch_var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            else:
                raise ValueError(f"Unexpected input dimensions: {x.dim()}")

            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
            self.running_var = 0.9 * self.running_var + 0.1 * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Add Gaussian noise to gamma
        if self.training:
            self.epsilon = torch.randn(self.num_features) * self.epsilon_std
        gamma_noisy = self.gamma * (1 + self.epsilon)

        # Normalize the input
        if x.dim() == 2:
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + 1e-5)
        elif x.dim() == 4:
            x_normalized = (x - batch_mean[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + 1e-5)

        return gamma_noisy * x_normalized + self.beta
