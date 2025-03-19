import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, num_classes, class_names=None):
    """Compute comprehensive classification metrics."""
    # Convert to numpy arrays if needed
    y_true = (
        y_true.cpu().numpy()
        if isinstance(y_true, torch.Tensor)
        else np.array(y_true)
    )
    y_pred = (
        y_pred.cpu().numpy()
        if isinstance(y_pred, torch.Tensor)
        else np.array(y_pred)
    )

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=np.arange(num_classes)
    )

    # Get class-wise accuracy from confusion matrix
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Create a dictionary with class names if provided
    class_metrics = {}
    for i in range(num_classes):
        class_name = (
            class_names[i]
            if class_names and i < len(class_names)
            else f"Class {i}"
        )
        class_metrics[class_name] = {
            "accuracy": class_accuracy[i] if i < len(class_accuracy) else 0,
            "precision": class_precision[i] if i < len(class_precision) else 0,
            "recall": class_recall[i] if i < len(class_recall) else 0,
            "f1": class_f1[i] if i < len(class_f1) else 0,
            "support": conf_matrix[:, i].sum(),
        }

    # Most confused pairs
    most_confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                # True class i predicted as class j
                confusion_value = conf_matrix[i, j]
                if confusion_value > 0:
                    class_i = (
                        class_names[i]
                        if class_names and i < len(class_names)
                        else f"Class {i}"
                    )
                    class_j = (
                        class_names[j]
                        if class_names and j < len(class_names)
                        else f"Class {j}"
                    )
                    most_confused_pairs.append(
                        (class_i, class_j, confusion_value)
                    )

    # Sort by confusion value
    most_confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "class_metrics": class_metrics,
        "most_confused_pairs": (
            most_confused_pairs[:5] if most_confused_pairs else []
        ),
    }


def print_metrics_report(metrics, class_names=None):
    """Print a formatted report of the metrics."""
    print("\n===== MODEL PERFORMANCE REPORT =====")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall: {metrics['recall']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_score']:.4f}")

    print("\n----- Per-Class Performance -----")
    print(
        f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
        f"{'F1-Score':<10} {'Support':<10}"
    )
    print("-" * 65)

    for class_name, metric in metrics["class_metrics"].items():
        print(
            f"{class_name:<15} {metric['accuracy']:.4f}     {metric['precision']:.4f}     "
            f"{metric['recall']:.4f}     {metric['f1']:.4f}     {metric['support']}"
        )

    print("\n----- Most Confused Class Pairs -----")
    print(f"{'True Class':<15} {'Predicted As':<15} {'Count':<10}")
    print("-" * 40)

    for true_class, pred_class, count in metrics["most_confused_pairs"]:
        print(f"{true_class:<15} {pred_class:<15} {count:<10}")
