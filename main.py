import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from torchsummary import summary
from torch.nn.functional import softmax
import torch.nn.functional as F
import netcal.metrics as metrics
from netcal.metrics import ECE
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from torch.utils.data import random_split
import multiprocessing
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import sys
# Get the current working directory of the notebook
current_dir = os.path.dirname(os.path.abspath('__file__'))
print(f"current dir is {current_dir}")
# Add the ABNN and 'Simple CNN Demo' directories to the Python path
abnn_dir = os.path.abspath(os.path.join(current_dir, './MakeMe-BNN/ABNN'))
print(f"ABNN dir is {abnn_dir}")
# if abnn_dir not in sys.path:
#     sys.path.insert(0, abnn_dir)

multiprocessing.set_start_method('forkserver', force=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True

# Now import the necessary modules
from ABNN import ABNN, test_and_eval, train
from ABNN.datasets import dtd,imagenet,cifar10,cifar100,streethazards,svhn,bddanomaly,muad
from ABNN.deep_learning_models import resnet,resnet_diff_arc,wide_resnet18_10,vit,deeplabv3plus_resnet50


def train_resnet50_on_cifar10(taindata, validdata, testdata):

    resnet50_cifr10 = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)
    resnet50_cifr10.to(device)
    summary(resnet50_cifr10, (3, 32, 32))

    # Training the model
    train_losses, val_losses = train.train_model(
        model=resnet50_cifr10, 
        train_loader=taindata, 
        val_loader=validdata,
        epochs=200, 
        learning_rate=0.1, 
        gamma_lr=0.2,
        milestones=[60, 120, 160], 
        save_path='./outputs/resnet50_cifr10_dropout.pth', 
        Weight_decay=5e-4,
        Momentum=0.9, 
        Optimizer_type='SGD',  
        Loss_fn='CrossEntropyLoss',
        Num_classes=10,
        BNL_enable=False,
        BNL_load_path='./outputs/resnet50_cifr10_dropout.pth'
    )

    # Testing the model with metrics
    test_and_eval.test_model_with_metrics(
        loss_fn=nn.CrossEntropyLoss(), 
        model=resnet50_cifr10, 
        test_loader=testdata, 
        load_path="./outputs/resnet50_cifr10_dropout.pth",
        calculate_uncert=True, 
        calculate_nll_loss=True, 
        calculate_ece_error=True,
        calculate_auprc=True, 
        calculate_auc_roc=True, 
        calculate_fpr_95=True, 
        count_params=True,
        plot_uncert=False, 
        predict_uncert=False, 
        model_class=resnet50_cifr10.__class__, 
        models=[torch.load('./outputs/resnet50_cifr10_dropout.pth')],
        num_samples=10, 
        num_classes=10
    )

    resnet50_cifr10_bnl = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10,norm_layer=ABNN.BNL)
    resnet50_cifr10_bnl.to(device)
    summary(resnet50_cifr10_bnl, (3, 32, 32))

    # Training the model
    train_losses, val_losses = train.train_model(
        model=resnet50_cifr10_bnl, 
        train_loader=taindata, 
        val_loader=validdata,
        epochs=200, 
        learning_rate=0.1, 
        gamma_lr=0.2,
        milestones=[60, 120, 160], 
        save_path='./outputs/resnet50_cifr10_bnl_dropout.pth', 
        Weight_decay=5e-4,
        Momentum=0.9, 
        Optimizer_type='SGD',  
        Loss_fn=ABNN.CustomMAPLoss,
        Num_classes=10,
        BNL_enable=True,
        BNL_load_path='./outputs/resnet50_cifr10_dropout.pth'
    )

    # Testing the model with metrics
    test_and_eval.test_model_with_metrics(
        loss_fn=nn.CrossEntropyLoss(), 
        model=resnet50_cifr10, 
        test_loader=testdata, 
        load_path="./outputs/resnet50_cifr10_bnl_dropout.pth",
        calculate_uncert=True, 
        calculate_nll_loss=True, 
        calculate_ece_error=True,
        calculate_auprc=True, 
        calculate_auc_roc=True, 
        calculate_fpr_95=True, 
        count_params=True,
        plot_uncert=False, 
        predict_uncert=False, 
        model_class=resnet50_cifr10.__class__, 
        models=[torch.load('./outputs/resnet50_cifr10_bnl_dropout.pth')],
        num_samples=10, 
        num_classes=10
    )


def main():
    # Load dataset
    trainloader_svhn, validloader_svhn, testloader_svhn = svhn.prepare_svhn_data()
    trainloader10, validloader10, testloader10 = cifar10.prepare_cifar10_data()
    trainloader100, validloader100, testloader100 = cifar100.prepare_cifar100_data()

    train_resnet50_on_cifar10(trainloader10, validloader10, testloader10)




if __name__ == "__main__":
    main()