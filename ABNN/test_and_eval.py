"""
test_and_eval.py

This file defines a test loop for evaluating the performance and uncertainty
of a neural network model. It is designed to calculate various metrics
as described in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian
Uncertainty from Pre-trained Models" (https://arxiv.org/abs/2312.15297).


### Description:

The `test_model_with_metrics` function evaluates a neural network model on a test dataset, providing flexibility to calculate various performance metrics and uncertainty measures based on the provided flags. This function is designed to offer a comprehensive evaluation with a single loop over the test data, ensuring efficient computation.

### Functionality:
1. Model Loading: Loads the model state from the specified path and sets the model to evaluation mode.
2. Single Test Loop: Iterates over the test dataset once to compute the required metrics.
4. Uncertainty Calculation: Computes the average uncertainty (variance) for each class if `calculate_uncert` is enabled.
5. Negative Log-Likelihood: Computes and prints the NLL if `calculate_nll_loss` is enabled.
6. Expected Calibration Error: Computes and prints the ECE if `calculate_ece_error` is enabled.
7. Precision-Recall AUC: Computes and prints the mean AUPR if `calculate_auprc` is enabled.
8. ROC AUC: Computes and prints the mean AUC if `calculate_auc_roc` is enabled.
9. FPR at 95% TPR: Computes and prints the mean FPR at 95% TPR if `calculate_fpr_95` is enabled.
10. Parameter Counting: Counts and prints the number of trainable parameters if `count_params` is enabled.
11. Uncertainty Plotting: Plots the uncertainty for different classes if `plot_uncert` is enabled.
12. Ensemble Prediction: Uses an ensemble of models for prediction if `predict_uncert` is enabled, calculating accuracy and variance.

This function ensures a flexible and efficient evaluation of the model, accommodating various metrics and uncertainty assessments as needed.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from ABNN.map import CustomMAPLoss
def test_model_with_metrics(loss_fn: nn.Module, model: nn.Module, test_loader: DataLoader, load_path: str = 'vit_mnist.pth',
               calculate_uncert: bool = False, calculate_nll_loss: bool = False, calculate_ece_error: bool = False,
               calculate_auprc: bool = False, calculate_auc_roc: bool = False, calculate_fpr_95: bool = False, 
               count_params: bool = False, plot_uncert: bool = False, predict_uncert: bool = False, 
               model_class: type = None, models: list = None, num_samples: int = 40, num_classes: int = 10,
               Weight_decay: float = 5e-4) -> None:
    """
    Evaluates the model on the test dataset with various metrics.

    Parameters:
        loss_fn (nn.Module): The loss function used while training.
        model (nn.Module): The neural network model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        load_path (str): Path to the file from which the model state is loaded.
        calculate_uncert (bool): Whether to calculate uncertainty.
        calculate_nll_loss (bool): Whether to calculate Negative Log-Likelihood.
        calculate_ece_error (bool): Whether to calculate Expected Calibration Error.
        calculate_auprc (bool): Whether to calculate Area Under the Precision-Recall Curve.
        calculate_auc_roc (bool): Whether to calculate Area Under the ROC Curve.
        calculate_fpr_95 (bool): Whether to calculate False Positive Rate at 95% True Positive Rate.
        count_params (bool): Whether to count the number of parameters.
        plot_uncert (bool): Whether to plot uncertainty.
        predict_uncert (bool): Whether to predict with uncertainty using ensemble.
        model_class (type): The model class for ensemble prediction.
        models (list): List of state dictionaries for ensemble prediction.
        num_samples (int): Number of Monte Carlo samples for uncertainty estimation.
        num_classes (int): Number of classes in the dataset.
        Weight_decay (float): Weight_decay of customMapLoss function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(load_path), strict=False)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    
    
    if loss_fn == "CustomMAPLoss":
        eta = torch.ones(num_classes)
        criterion = CustomMAPLoss(eta, model.parameters()).to(device)    
    else:
        criterion = loss_fn
    correct = 0
    total = 0
    test_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    uncertainties = {i: [] for i in range(num_classes)} if calculate_uncert else None

    # Single loop to compute metrics
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Calculate uncertainty if required
            if calculate_uncert:
                mc_outputs = torch.stack([model(images) for _ in range(num_samples)])
                probabilities = torch.softmax(mc_outputs, dim=-1)
                variance = probabilities.var(dim=0)
                for i in range(num_classes):
                    class_mask = (labels == i)
                    if class_mask.any():
                        class_variance = variance[class_mask, i].mean().item()
                        uncertainties[i].append(class_variance)

            # Store probabilities for further metrics
            if calculate_ece_error or calculate_auprc or calculate_auc_roc or calculate_fpr_95:
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

    # Calculate and print metrics
    accuracy = 100 * correct / total 
    avg_uncertainties = {i: np.mean(uncertainties[i]) if uncertainties[i] else 0 for i in range(num_classes)} if calculate_uncert else None
    test_loss /= len(test_loader)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else None
    all_targets = np.array(all_targets)
    ece_score = ECE(bins=10).measure(all_probs, all_targets) if calculate_ece_error else None
    
    if calculate_auprc:
        auprs = []
        all_labels_one_hot = np.eye(num_classes)[all_targets]
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(all_labels_one_hot[:, i], all_probs[:, i])
            aupr = auc(recall, precision)
            auprs.append(aupr)
        mean_aupr = np.mean(auprs)
    else:
        mean_aupr = None
    
    if calculate_auc_roc:
        aucs = []
        all_labels_one_hot = np.eye(num_classes)[all_targets]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        mean_auc = np.mean(aucs)
    else:
        mean_auc = None
    
    if calculate_fpr_95:
        fpr_95_recall = []
        all_labels_one_hot = np.eye(num_classes)[all_targets]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], all_probs[:, i])
            idx = np.where(tpr >= 0.95)[0][0]
            fpr_at_95_recall = fpr[idx]
            fpr_95_recall.append(fpr_at_95_recall)
        mean_fpr_95_recall = np.mean(fpr_95_recall)
    else:
        mean_fpr_95_recall = None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) if count_params else None
    
    if predict_uncert:
        if model_class is None or models is None:
            raise ValueError("model_class and models must be provided for uncertainty prediction.")
        
        ensemble_outputs = []
        all_labels = []

        for model_state_dict in models:
            net = model_class()
            net.load_state_dict(model_state_dict)
            net.to(device)
            net.eval()
            
            with torch.no_grad():
                all_outputs = []
                for _ in range(num_samples):
                    batch_outputs = []
                    batch_labels = []
                    for data in test_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
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

        ensemble_outputs = torch.stack(ensemble_outputs).mean(0)
        _, predicted = torch.max(ensemble_outputs, 1)
        
        correct = (predicted.cpu().numpy() == all_labels).sum()
        total = len(all_labels)
        accuracy = 100 * correct / total
        
        class_variances = {}
        for class_id in range(num_classes):
            class_mask = (all_labels == class_id)
            if class_mask.any():
                class_predictions = ensemble_outputs[class_mask, class_id].cpu().numpy()
                class_variance = np.var(class_predictions)
                class_variances[class_id] = class_variance
                print(f'Class {class_id} variance: {class_variance:.6f}')
        
        if plot_uncert:
            fig, ax = plt.subplots(figsize=(10, 6))
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

    # Print results
    print(f'Test set Metrics:\n Average loss: {test_loss:.4f} \n F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    if calculate_uncert:
        print(f'Uncertainties: {avg_uncertainties}')
    if calculate_ece_error:
        print(f'ECE: {ece_score:.4f}')
    if calculate_auprc:
        print(f'Mean AUPR: {mean_aupr:.4f}')
    if calculate_auc_roc:
        print(f'Mean AUC: {mean_auc:.4f}')
    if calculate_fpr_95:
        print(f'Mean FPR at 95% Recall: {mean_fpr_95_recall:.4f}')
    if count_params:
        print(f'Number of Parameters: {param_count}')