import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

def prepare_svhn_data(batch_size=128, train_split_ratio=0.8, num_workers=2):
    """
    Prepares the SVHN dataset for training, validation, and testing.

    Parameters:
    - batch_size (int): The number of samples per batch to load. Default is 128.
    - train_split_ratio (float): The proportion of the dataset to include in the train split. Default is 0.8 (80%).
    - num_workers (int): How many subprocesses to use for data loading. Default is 2.

    Returns:
    - trainloader (DataLoader): DataLoader for the training set.
    - validloader (DataLoader): DataLoader for the validation set.
    - testloader (DataLoader): DataLoader for the test set.
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the SVHN dataset
    full_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(train_split_ratio * len(full_trainset))  # 80% for training
    valid_size = len(full_trainset) - train_size  # 20% for validation
    train_subset, valid_subset = random_split(full_trainset, [train_size, valid_size])

    # Create DataLoaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load the test set
    testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print the sizes of the datasets
    print(f'SVHN Training set size: {len(train_subset)}')
    print(f'SVHN Validation set size: {len(valid_subset)}')
    print(f'SVHN Test set size: {len(testset)}')

    return trainloader, validloader, testloader

if __name__ == "__main__":
    trainloader_svhn, validloader_svhn, testloader_svhn = prepare_svhn_data()
