"""
ABNN_Metrics.py

This file defines a set of functions for evaluating the performance and uncertainty
of a neural network model. These functions are designed to calculate various metrics
as described in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian
Uncertainty from Pre-trained Models" (https://arxiv.org/abs/2312.15297).

Functions:
    calculate_accuracy: Calculates the accuracy of the model on a given dataset.
    calculate_uncertainty: Calculates the average uncertainty (variance) for each class.
    calculate_nll: Calculates the Negative Log-Likelihood (NLL) of the model.
    calculate_ece: Calculates the Expected Calibration Error (ECE) of the model.
    calculate_aupr: Calculates the Area Under the Precision-Recall Curve (AUPR) for each class.
    calculate_auc: Calculates the Area Under the ROC Curve (AUC) for each class.
    calculate_fpr95: Calculates the False Positive Rate at 95% True Positive Rate (FPR95).
    count_parameters: Counts the number of trainable parameters in the model.
    predict_with_uncertainty: Predicts with an ensemble of models and calculates accuracy and variance.
    plot_uncertainty: Plots the uncertainty (variance) of predictions for different classes.
"""

import torch
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from netcal.metrics import ECE

def calculate_accuracy(loader, model):
    """
    Calculate the Accuracy (Acc) of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy

def calculate_uncertainty(loader, model, num_samples=10, num_classes=10):
    """
    Calculate the Uncertainty of the each class.
    """
    model.eval()
    class_variances = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = [model(images) for _ in range(num_samples)]
            outputs = torch.stack(outputs)
            probabilities = torch.softmax(outputs, dim=-1)
            variance = probabilities.var(dim=0)
            for i in range(num_classes):
                class_mask = (labels == i)
                if class_mask.any():
                    class_variance = variance[class_mask, i].mean().item()
                    class_variances[i].append(class_variance)
    avg_class_variances = {i: np.mean(class_variances[i]) if class_variances[i] else 0 for i in range(num_classes)}
    for class_id, uncertainty in avg_class_variances.items():
        print(f'Average variance for class {class_id}: {uncertainty:.6f}')
    return avg_class_variances

def calculate_nll(loader, model, criterion):
    """
    Calculate the Negative Log-Likelihood (NLL) of the model.
    """
    nll_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels, model)
            nll_loss += loss.item()
    nll_loss /= len(loader)
    print(f'Negative Log-Likelihood: {nll_loss:.4f}')
    return nll_loss


def calculate_ece(loader, model, bins=10):
    """
    Calculate the Expected Calibration Error (ECE) of the model.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    ece = ECE(bins=bins)
    ece_score = ece.measure(all_probs, all_labels)
    print(f'Expected Calibration Error (ECE): {ece_score:.4f}')
    return ece_score

def calculate_aupr(loader, model):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPR) for each class.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    n_classes = all_probs.shape[1]
    all_labels_one_hot = np.eye(n_classes)[all_labels]
    auprs = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(all_labels_one_hot[:, i], all_probs[:, i])
        aupr = auc(recall, precision)
        auprs.append(aupr)
        print(f'Class {i}: AUPR = {aupr:.4f}')
    mean_aupr = np.mean(auprs)
    print(f'Mean AUPR: {mean_aupr:.4f}')
    return mean_aupr

def calculate_auc(loader, model):
    """
    Calculate the Area Under the ROC Curve (AUC) for each class.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    n_classes = all_probs.shape[1]
    all_labels_one_hot = np.eye(n_classes)[all_labels]
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print(f'Class {i}: AUC = {roc_auc:.4f}')
    mean_auc = np.mean(aucs)
    print(f'Mean AUC: {mean_auc:.4f}')
    return mean_auc

def calculate_fpr95(loader, model):
    """
    Calculate the False Positive Rate at 95% True Positive Rate (FPR95) for each class.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    n_classes = all_probs.shape[1]
    all_labels_one_hot = np.eye(n_classes)[all_labels]
    fpr_95_recall = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
        idx = np.where(tpr >= 0.95)[0][0]
        fpr_at_95_recall = fpr[idx]
        fpr_95_recall.append(fpr_at_95_recall)
        print(f'Class {i}: FPR at 95% Recall = {fpr_at_95_recall:.4f}')
    mean_fpr_95_recall = np.mean(fpr_95_recall)
    print(f'Mean FPR at 95% Recall: {mean_fpr_95_recall:.4f}')
    return mean_fpr_95_recall

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of Parameters: {param_count}')
    return param_count

def predict_with_uncertainty(model_class, models, testloader, num_samples=10, num_classes=10):
    """
    Predict with an ensemble of models and calculate accuracy and variance.
    
    Parameters:
    - model_class: The class of the model to be instantiated.
    - models: List of state dictionaries for the models.
    - testloader: DataLoader for the test dataset.
    - num_samples: Number of Monte Carlo samples to draw for uncertainty estimation.
    - num_classes: Number of classes in the dataset.
    
    Returns:
    - ensemble_outputs: The mean of the ensemble outputs.
    - all_labels: Ground truth labels for the test set.
    - predicted_classes: Predicted classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_outputs = []
    all_labels = []
    
    M = len(models)  # Number of models (ensemble members)
    
    for model_state_dict in models:
        net = model_class()  # Instantiate the model
        net.load_state_dict(model_state_dict)
        net.to(device)
        net.eval()
        
        with torch.no_grad():
            all_outputs = []
            for _ in range(num_samples):
                batch_outputs = []
                batch_labels = []
                for data in testloader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    noise = torch.randn(inputs.shape[0], net.fc3.out_features).to(device)  # Sample from Gaussian
                    outputs = net(inputs)
                    outputs = softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    batch_outputs.append(outputs + noise)
                    batch_labels.append(labels)
                all_outputs.append(torch.cat(batch_outputs))
                if len(all_labels) == 0:
                    all_labels = torch.cat(batch_labels).cpu().numpy()
            all_outputs = torch.stack(all_outputs).mean(0)
            ensemble_outputs.append(all_outputs)
    
    # Average the outputs of all ensemble members
    ensemble_outputs = torch.stack(ensemble_outputs).mean(0)
    _, predicted = torch.max(ensemble_outputs, 1)
    
    # Calculate accuracy
    correct = (predicted.cpu().numpy() == all_labels).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    # Calculate class variances
    class_variances = {}
    for class_id in range(num_classes):
        class_mask = (all_labels == class_id)
        if class_mask.any():
            class_predictions = ensemble_outputs[class_mask, class_id].cpu().numpy()
            class_variance = np.var(class_predictions)
            class_variances[class_id] = class_variance
            print(f'Class {class_id} variance: {class_variance:.6f}')
    
    return ensemble_outputs.cpu().numpy(), all_labels, predicted.cpu().numpy()

def plot_uncertainty(predictions, labels, num_classes=10):
    """
    Plot the uncertainty (variance) of predictions for different classes.
    
    Parameters:
    - predictions: Predicted probabilities from the model.
    - labels: Ground truth labels for the test set.
    - num_classes: Number of classes in the dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    class_variances = {i: [] for i in range(num_classes)}
    for i in range(len(predictions)):
        predicted_probabilities = predictions[i]
        true_label = labels[i]
        class_variances[true_label].append(predicted_probabilities)

    for class_id in range(num_classes):
        if class_variances[class_id]:
            class_predictions = np.array(class_variances[class_id])
            class_variance = class_predictions.var(axis=0)
            class_mean = class_predictions.mean(axis=0)
            ax.errorbar(class_id, class_mean[class_id], yerr=class_variance[class_id], fmt='o', label=f'Class {class_id}')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Uncertainty in Predictions')
    ax.legend(loc='upper right')
    plt.show()