"""
### Training Loop:

The `train_model` function is a versatile training loop designed for PyTorch models, providing flexibility in selecting optimizers, loss functions, and various hyperparameters. Here's a brief description of its key features and how it utilizes different parameters:

1. Optimizer Selection:
   - The function allows the choice between 'SGD' (Stochastic Gradient Descent) and 'Adam' optimizers via the `Optimizer_type` parameter. This is achieved by checking the value of `Optimizer_type` and initializing the respective optimizer with the specified `learning_rate`, `Weight_decay`, and `Momentum` (for SGD).

2. Loss Function Selection:
   - The function supports multiple loss functions, including 'CrossEntropyLoss', 'MSELoss', and a custom loss function 'CustomMAPLoss'. The appropriate loss function is selected based on the `Loss_fn` parameter. For 'CustomMAPLoss', the `Num_classes` and `Weight_decay` parameters are used for initialization.

3. Learning Rate Scheduler:
   - A MultiStepLR scheduler is used to adjust the learning rate at specified milestones. The `milestones` parameter defines the epochs at which the learning rate is reduced by a factor specified by `gamma_lr`.

4. Training and Validation Loop:
   - The function contains a standard training loop where it iterates over the training dataset, computes the loss, performs backpropagation, and updates the model parameters.
   - After each epoch, the model is evaluated on the validation dataset, and the average validation loss is computed and stored.

5. Hyperparameters:
   - Various hyperparameters such as `epochs`, `learning_rate`, `Weight_decay`, `Momentum`, and `Num_classes` can be adjusted to fine-tune the training process according to specific needs.

6. Model Saving:
   - The trained model's state dictionary is saved to a specified path (`save_path`) after the training is complete.

7. Loss Visualization:
   - The function plots the training and validation losses over epochs for easy visualization of the model's performance.


This design provides flexibility and ease of experimentation with different training configurations, making it suitable for various deep learning tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ABNN.map import CustomMAPLoss, ABNNLoss
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 10, learning_rate: float = 0.005, gamma_lr: float = 0.1, Scheduler = True, 
                milestones: list = [5, 15], save_path: str = 'model.pth', Weight_decay: float = 5e-4,
                Momentum: float = 0.9, Optimizer_type: str = 'SGD',  Loss_fn: str = 'CrossEntropyLoss',
                Num_classes: int = 10, BNL_enable: bool = False, BNL_load_path: str = "model.pth") -> (list, list):
    """
    Trains the model and evaluates it on the validation set after each epoch.

    Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the optimizer.
        gamma_lr (float): Factor by which the learning rate will be multiplied at each milestone.
        milestones (list): List of epoch indices at which to adjust the learning rate.
        save_path (str): Path to save the trained model state.
        Weight_decay (float): The weight decay for the optimizer.
        Momentum (float): The Momentum for the optimizer.
        Optimizer_type (str): The optimizer type.
        Loss_fn (str): The loss function type.
        Num_classes (int): Number of classes in the dataset.
        BNL_enable (bool): this to enable the training loop to load the trined wights of deep learning model.
        BNL_load_path (str): the loading path of the trined wights of deep learning model.


    Returns:
        train_losses (tuple): A tuple containing lists of training losses per epoch.
        val_losses (tuple): A tuple containing lists of validation losses per epoch.
    """  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    if BNL_enable:
        model.load_state_dict(torch.load(BNL_load_path),strict=False)
        print("BNL model loaded from {}".format(BNL_load_path))
        print('Model weights loaded.')

    if Optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=Momentum, weight_decay=Weight_decay)
    elif Optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Weight_decay)
    else:
        raise ValueError("Unsupported optimizer type. Choose either 'SGD' or 'Adam'.")

    if Loss_fn == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif Loss_fn == 'MSELoss':
        criterion = nn.MSELoss()
    elif Loss_fn == 'CustomMAPLoss':
        eta = torch.ones(Num_classes)
        criterion = CustomMAPLoss(eta, model.parameters()).to(device)    
    elif Loss_fn == 'ABNNLoss':
        criterion = ABNNLoss(Num_classes, model.parameters(), Weight_decay).to(device)    
    else:
        raise ValueError("Unsupported loss function. Implement additional loss functions as needed. Choose either 'CrossEntropyLoss' or 'MSELoss' or 'CustomMAPLoss'")
    

    if Scheduler == True:
       scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma_lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0

        # Training loop
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Loss calculation
            loss.backward()  # Backward pass (backpropagation)
            optimizer.step()  # Optimize model parameters
            train_loss += loss.item()

        # Store average training loss
        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        val_loss = 0.0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        # Store average validation loss
        val_losses.append(val_loss / len(val_loader))

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

        if Scheduler == True:
           scheduler.step()  # Adjust learning rate
           
        # Empty the memory Periodically
        torch.cuda.empty_cache()

    # Save the trained model state
    torch.save(model.state_dict(), save_path)

    # Plot training and validation losses
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses
