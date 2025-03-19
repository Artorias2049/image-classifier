import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from numpy import dtype
from torch.utils.tensorboard import SummaryWriter

from models.models import VehicleClassifier
from preprocessing.clean_data import clean_dataset
from preprocessing.preprocessor import preprocess_dataset
from utils.dataloader import get_dataloaders
from utils.metrics import compute_metrics, print_metrics_report
from utils.training_plots import (
    plot_confusion_matrix,
    plot_lr_schedule,
    plot_per_class_accuracy,
    save_training_plots,
)

torch.set_default_dtype(torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a vehicle classification model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b0",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "mobilenet_v3_small",
        ],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.6, help="Dropout rate"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.2,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--no-mixup", action="store_true", help="Disable mixup augmentation"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Skip preprocessing steps"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="strong",
        choices=["none", "light", "medium", "strong"],
        help="Augmentation strength",
    )
    parser.add_argument(
        "--image-size", type=int, default=244, help="Target image size"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./vehicle_dataset",
        help="Path to input dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cleaned_dataset",
        help="Path to output dataset",
    )
    parser.add_argument(
        "--logs-dir", type=str, default="./train_logs", help="Path to save logs"
    )
    parser.add_argument(
        "--plots-dir", type=str, default="./plots", help="Path to save plots"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    return parser.parse_args()


def setup_directories(*dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        if "logs_dir" in dir_path:
            os.makedirs(os.path.join(dir_path, "tensorboard"), exist_ok=True)


def calculate_loss(outputs, labels, mixed_labels=None, criterion=None):
    """Calculate the loss with optional mixup."""
    if mixed_labels is None:
        return criterion(outputs, labels)
    else:
        labels_a, labels_b, lam = mixed_labels
        return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
            outputs, labels_b
        )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    mixup_fn=None,
    warmup_scheduler=None,
    main_scheduler=None,
    warmup_epochs=5,
    scaler=None,
):
    """Train the model for one epoch with optional mixed precision and mixup."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    learning_rates = []
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Force images to be float32
        if images.dtype != torch.float32:
            images = images.float()

        images = images.to(device)
        labels = labels.to(device)

        # Apply mixup if enabled
        mixed_labels = None
        if mixup_fn is not None:
            images, labels_a, labels_b, lam = mixup_fn((images, labels))
            mixed_labels = (labels_a, labels_b, lam)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass, loss calculation, and backward pass
        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = calculate_loss(outputs, labels, mixed_labels, criterion)

            # Make sure loss is a scalar
            if hasattr(loss, "shape") and loss.shape != torch.Size([]):
                # If loss is not a scalar, reduce it
                loss = loss.mean()

            # Scale and perform backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(images)
            loss = calculate_loss(outputs, labels, mixed_labels, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update learning rate using the appropriate scheduler
        if epoch < warmup_epochs and warmup_scheduler:
            warmup_scheduler.step()
        elif main_scheduler:
            main_scheduler.step()

        # Track learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0) if mixed_labels is None else labels_a.size(0)

        # Calculate accuracy (use original labels for mixup)
        if mixed_labels is None:
            correct += (predicted == labels).sum().item()
        else:
            correct += (predicted == labels_a).sum().item()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )

    # Calculate metrics
    avg_loss = running_loss / len(train_loader)
    avg_acc = correct / total
    print(f"Epoch training time: {time.time() - start_time:.2f} seconds")

    return avg_loss, avg_acc, learning_rates


def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    class_names=None,
    writer=None,
    num_classes=None,
):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    class_correct, class_total = {}, {}

    with torch.no_grad():
        for images, labels in val_loader:
            # Force images to be float32
            if images.dtype != torch.float32:
                images = images.float()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Track per-class accuracy
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0

                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total

    # Calculate per-class accuracy
    per_class_acc = {
        cls: class_correct[cls] / class_total[cls] for cls in class_correct
    }

    # Log metrics to tensorboard
    if writer:
        for cls, acc in per_class_acc.items():
            writer.add_scalar(f"Accuracy/val_class_{cls}", acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Print per-class accuracy
    for cls, acc in per_class_acc.items():
        cls_name = (
            class_names[cls]
            if class_names and cls < len(class_names)
            else str(cls)
        )
        print(f"Class {cls_name} accuracy: {acc:.4f}")

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, num_classes, class_names)

    if writer:
        writer.add_scalar("Precision/val", metrics["precision"], epoch)
        writer.add_scalar("Recall/val", metrics["recall"], epoch)
        writer.add_scalar("F1/val", metrics["f1_score"], epoch)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    # Save plots if class names are provided
    if class_names:
        plot_confusion_matrix(
            all_labels, all_preds, class_names, output_dir="plots"
        )
        plot_per_class_accuracy(
            class_correct, class_total, class_names, output_dir="plots"
        )

    return val_loss, val_acc, metrics


def main():
    args = parse_args()
    setup_directories(args.logs_dir, args.plots_dir)

    # Set up tensorboard logging
    writer = SummaryWriter(os.path.join(args.logs_dir, "tensorboard"))

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print(f"Training on device: {device}")

    # Hyperparameters
    num_classes = 12
    freeze_layers = True
    dropout_rate = args.dropout_rate
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    patience = args.patience
    warmup_epochs = args.warmup_epochs
    model_name = args.model_name
    label_smoothing = args.label_smoothing
    use_mixup = not args.no_mixup
    aug_strength = args.augmentation
    image_size = args.image_size

    # Preprocess images if needed
    if not args.no_preprocess:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("Preprocessing images...")
        preprocess_dataset(
            args.input_dir,
            output_dir,
            target_size=(image_size, image_size),
            num_workers=args.num_workers,
        )
        print("Cleaning dataset...")
        clean_dataset(output_dir, os.path.join(output_dir, "log_reports/"))

    # Get class names
    train_dir = os.path.join(args.output_dir, "train")
    class_names = sorted(
        [
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
    )
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    # Load cleaned dataset
    print("Loading dataloaders...")
    train_loader, val_loader, mixup_fn = get_dataloaders(
        train_dir,
        os.path.join(args.output_dir, "val"),
        batch_size=batch_size,
        num_workers=args.num_workers,
        use_mixup=use_mixup,
        image_size=image_size,
        augmentation_strength=aug_strength,
    )

    if not use_mixup:
        mixup_fn = None

    # Model setup
    print(f"Creating model: {model_name}")
    model = VehicleClassifier(
        num_classes=num_classes,
        pretrained=True,
        freeze_layers=freeze_layers,
        dropout_rate=dropout_rate,
        model_name=model_name,
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing, reduction="mean"
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Schedulers
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader),
    )

    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs) * len(train_loader),
        eta_min=learning_rate / 100,
    )

    # Track training metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    all_learning_rates = []

    # Mixed precision training for speed
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop
    best_val_acc, best_val_f1 = 0.0, 0.0
    save_path = os.path.join(args.logs_dir, "model_best.pth")
    save_path_f1 = os.path.join(args.logs_dir, "model_best_f1.pth")
    stagnant_epochs, layers_unfrozen = 0, False

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Unfreeze layers after warmup
        if epoch == warmup_epochs and not layers_unfrozen and freeze_layers:
            print("Unfreezing layers for fine-tuning")
            model.unfreeze_layers()
            layers_unfrozen = True

        # Train one epoch
        train_loss, train_acc, epoch_learning_rates = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            mixup_fn,
            warmup_scheduler,
            main_scheduler,
            warmup_epochs,
            scaler,
        )

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Extend learning rates list
        all_learning_rates.extend(epoch_learning_rates)

        # Validate
        val_loss, val_acc, metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            class_names=class_names,
            writer=writer,
            num_classes=num_classes,
        )

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save models based on different metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stagnant_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
                save_path,
            )
            print(f"Best model saved with accuracy: {best_val_acc:.4f}")
        else:
            stagnant_epochs += 1

        # Save based on F1 score as well
        if metrics["f1_score"] > best_val_f1:
            best_val_f1 = metrics["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": metrics["f1_score"],
                },
                save_path_f1,
            )
            print(f"Best F1 model saved with F1: {best_val_f1:.4f}")

        if stagnant_epochs >= patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs. Training terminated."
            )
            break

    # Close tensorboard writer
    writer.close()

    # Save plots
    print("Generating training plots...")
    save_training_plots(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        output_dir=args.plots_dir,
    )

    # Plot learning rate schedule
    plot_lr_schedule(all_learning_rates, output_dir=args.plots_dir)

    # Load and evaluate best model
    print("\nEvaluating best model on validation set:")
    best_model = VehicleClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_layers=False,
        dropout_rate=0.0,
        model_name=model_name,
    ).to(device)

    checkpoint = torch.load(save_path)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    # Final validation and generate plots
    final_val_loss, final_val_acc, final_metrics = validate(
        best_model,
        val_loader,
        criterion,
        device,
        num_epochs - 1,
        class_names=class_names,
        num_classes=num_classes,
    )

    print(f"Best model validation accuracy: {final_val_acc:.4f}")
    print(f"Best model F1-score: {final_metrics['f1_score']:.4f}")

    # Print detailed metrics report
    print_metrics_report(final_metrics, class_names)

    # Log the results to a text file for future reference
    with open(os.path.join(args.plots_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Best model validation accuracy: {final_val_acc:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"Recall: {final_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {final_metrics['f1_score']:.4f}\n")
        f.write("\nPer-class accuracy:\n")

        # Get class-wise performance
        class_correct, class_total = {}, {}
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = best_model(images)
                _, predicted = torch.max(outputs, 1)

                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    if label_item not in class_correct:
                        class_correct[label_item] = 0
                        class_total[label_item] = 0

                    class_total[label_item] += 1
                    if label_item == pred.item():
                        class_correct[label_item] += 1

        # Write per-class accuracy
        for cls in sorted(class_correct.keys()):
            cls_name = (
                class_names[cls] if cls < len(class_names) else f"Unknown_{cls}"
            )
            acc = class_correct[cls] / class_total[cls]
            f.write(f"Class {cls_name}: {acc:.4f}\n")

    print("All plots and reports saved to the plots directory")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
