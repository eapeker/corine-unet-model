import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from .data import corine_classes


def calculate_class_performance_metrics(true_labels, predicted_labels, corine_classes):
    true_class_indices = np.argmax(true_labels, axis=-1).flatten()
    predicted_class_indices = np.argmax(predicted_labels, axis=-1).flatten()

    iou_scores = []
    class_accuracy_scores = []
    class_names = []

    for i in range(true_labels.shape[-1]):
        true_mask = (true_class_indices == i)
        predicted_mask = (predicted_class_indices == i)

        intersection = np.logical_and(true_mask, predicted_mask).sum()
        union = np.logical_or(true_mask, predicted_mask).sum()

        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)

        class_accuracy = accuracy_score(
            true_class_indices[true_mask],
            predicted_class_indices[true_mask]
        ) if np.sum(true_mask) > 0 else 0
        class_accuracy_scores.append(class_accuracy)

        class_names.append(list(corine_classes.values())[i])

    performance_df = pd.DataFrame({
        'Class': class_names,
        'IoU': iou_scores,
        'Class Accuracy': class_accuracy_scores
    })

    return performance_df


def calculate_test_metrics(y_test, y_pred):
    """
    Calculate class-wise performance metrics for CORINE Level 2 test data

    Parameters:
        y_test: Ground truth labels
        y_pred: Model predictions
    """
    # CORINE Level 2 class names
    class_names = list(corine_classes.values())

    # Convert from one-hot to class indices
    y_true = np.argmax(y_test, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    # Calculate metrics
    iou_scores = []
    accuracy_scores = []

    for i in range(len(class_names)):
        # Calculate IoU
        true_mask = (y_true == i)
        pred_mask = (y_pred == i)
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)

        # Calculate Class Accuracy
        true_positive = intersection
        total_true = true_mask.sum()
        accuracy = true_positive / total_true if total_true > 0 else 0
        accuracy_scores.append(accuracy)

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'IoU': iou_scores,
        'Class Accuracy': accuracy_scores
    })

    return metrics_df


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix (%)", save_path=None):
    """
    Visualize and optionally save the normalized confusion matrix for each class.

    Parameters:
    - y_test: Ground truth labels (one-hot encoded)
    - y_pred: Predicted labels (one-hot encoded)
    - title: Plot title
    - save_path: Path to save the figure
    """
    # Convert from one-hot to class indices
    y_true_flat = np.argmax(y_test, axis=-1).flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()

    # Number of classes
    n_classes = y_test.shape[-1]

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(n_classes))

    # Normalize the confusion matrix to percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percentage = np.nan_to_num(cm_percentage)  # Replace NaN with 0

    # Class names for CORINE Level 2
    class_names = [
        'Urban', 'Industrial', 'Mine', 'GreenArea', 'Arable',
        'PermanentCrop', 'Pasture', 'MixedAgri', 'Forest', 'Shrub',
        'OpenSpace', 'InlandWet', 'CoastalWet', 'InlandWater', 'Marine'
    ]

    # Visualization
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save if path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return cm_percentage
