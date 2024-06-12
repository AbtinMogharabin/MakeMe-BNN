from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class BDDAnomaly(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(720, 1280)]
        self.ood_id = [1]

        for im in os.listdir(os.path.join(self.root, 'images')):
            self.images.append(os.path.join(self.root, 'images', im))
            self.img_labels.append(os.path.join(self.root, 'train_id', im.replace('.jpg', '_train_id.png')))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
    


def prepare_bddanomaly_data(root, batch_size=128, train_split_ratio=0.8, num_workers=2):
    """
    Prepares the BDDAnomaly dataset for training, validation, and testing.

    Parameters:
    - root (str): The root directory of the BDDAnomaly dataset.
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
        transforms.RandomCrop(128, padding=4),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the BDDAnomaly dataset
    full_dataset = BDDAnomaly(root=root)
    full_dataset.transform = transform

    # Split the dataset into training and validation sets
    train_size = int(train_split_ratio * len(full_dataset))  # 80% for training
    valid_size = len(full_dataset) - train_size  # 20% for validation
    train_subset, valid_subset = random_split(full_dataset, [train_size, valid_size])

    # Create DataLoaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Assuming the test set is a separate split within the BDDAnomaly dataset structure
    testset = BDDAnomaly(root=root)
    testset.transform = transform
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print the sizes of the datasets
    print(f'BDDAnomaly Training set size: {len(train_subset)}')
    print(f'BDDAnomaly Validation set size: {len(valid_subset)}')
    print(f'BDDAnomaly Test set size: {len(testset)}')

    return trainloader, validloader, testloader

if __name__ == "__main__":
    trainloader_bdd, validloader_bdd, testloader_bdd = prepare_bddanomaly_data(root='./bdd_data')
