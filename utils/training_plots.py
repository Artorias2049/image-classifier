import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Global setting to prevent plots from being displayed interactively
plt.switch_backend("agg")


def save_training_plots(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    output_dir="plots",
):
    """Save training and validation loss/accuracy plots"""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Loss plot
    axes[0].plot(
        epochs, train_losses, "b-", label="Train Loss", marker="o", markersize=3
    )
    axes[0].plot(
        epochs,
        val_losses,
        "r-",
        label="Validation Loss",
        marker="o",
        markersize=3,
    )
    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_title("Training & Validation Loss", fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Accuracy plot
    axes[1].plot(
        epochs,
        train_accuracies,
        "b-",
        label="Train Accuracy",
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        epochs,
        val_accuracies,
        "r-",
        label="Validation Accuracy",
        marker="o",
        markersize=3,
    )
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_title("Training & Validation Accuracy", fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_plots.png"), dpi=300)
    plt.close(fig)

    # Save also as individual plots for reference
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        epochs, train_losses, "b-", label="Train Loss", marker="o", markersize=3
    )
    plt.plot(
        epochs,
        val_losses,
        "r-",
        label="Validation Loss",
        marker="o",
        markersize=3,
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training & Validation Loss", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        train_accuracies,
        "b-",
        label="Train Accuracy",
        marker="o",
        markersize=3,
    )
    plt.plot(
        epochs,
        val_accuracies,
        "r-",
        label="Validation Accuracy",
        marker="o",
        markersize=3,
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Training & Validation Accuracy", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    all_labels, all_preds, class_names, output_dir="plots"
):
    """Plot and save confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


def plot_per_class_accuracy(
    class_correct, class_total, class_names, output_dir="plots"
):
    """Plot and save per-class accuracy"""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate per-class accuracy
    accuracies = {
        cls_idx: class_correct[cls_idx] / class_total[cls_idx]
        for cls_idx in range(len(class_names))
        if cls_idx in class_total and class_total[cls_idx] > 0
    }

    # Convert to lists for plotting
    classes = [class_names[idx] for idx in accuracies.keys()]
    acc_values = list(accuracies.values())

    # Sort by accuracy
    sorted_indices = np.argsort(acc_values)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_acc = [acc_values[i] for i in sorted_indices]

    # Plot
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_classes, sorted_acc, color="skyblue")

    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{sorted_acc[i]:.1%}",
            va="center",
            fontsize=10,
        )

    plt.xlabel("Accuracy", fontsize=14)
    plt.ylabel("Class", fontsize=14)
    plt.title("Per-Class Accuracy", fontsize=16)
    plt.xlim(0, 1.1)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=300)
    plt.close(fig)


def plot_lr_schedule(learning_rates, output_dir="plots"):
    """Plot and save learning rate schedule"""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, "b-")
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Learning Rate", fontsize=14)
    plt.title("Learning Rate Schedule", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_schedule.png"), dpi=300)
    plt.close(fig)
