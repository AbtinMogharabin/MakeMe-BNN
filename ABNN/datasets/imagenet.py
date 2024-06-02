import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader


def prepare_dtd_data(data_dir, batch_size=128, train_split_ratio=0.8, num_workers=2):
    """
    Prepares the Describable Textures Dataset (DTD) for training, validation, and testing.

    Parameters:
    - data_dir (str): Directory containing the DTD data.
    - batch_size (int): The number of samples per batch to load. Default is 128.
    - train_split_ratio (float): The proportion of the dataset to include in the train split. Default is 0.8 (80%).
    - num_workers (int): How many subprocesses to use for data loading. Default is 2.

    Returns:
    - trainloader (DataLoader): DataLoader for the training set.
    - validloader (DataLoader): DataLoader for the validation set.
    - testloader (DataLoader): DataLoader for the test set.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.DTD(root=data_dir, split='train', download=True, transform=transform)
    test_dataset = datasets.DTD(root=data_dir, split='test', download=True, transform=transform)

    train_size = int(train_split_ratio * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_subset, valid_subset = random_split(full_dataset, [train_size, valid_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'DTD Training set size: {len(train_subset)}')
    print(f'DTD Validation set size: {len(valid_subset)}')
    print(f'DTD Test set size: {len(test_dataset)}')

    return trainloader, validloader, testloader

if __name__ == "__main__":
    dtd_dir = 'ABNN/datasets/dtd-r1.0.1.tar.gz'

    # Load DTD data
    train_loader_dtd, val_loader_dtd, test_loader_dtd = prepare_dtd_data(dtd_dir, batch_size=32, train_split_ratio=0.8)
