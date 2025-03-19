import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.models import VehicleClassifier


def load_model(model_path, model_name, num_classes, device=None):
    """Load model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VehicleClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_layers=False,
        dropout_rate=0.0,  # No dropout for inference
        model_name=model_name,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded model from {model_path} (acc: {checkpoint.get('val_acc', 'N/A'):.4f})"
    )
    return model, device


def get_transforms(image_size=244, augment=False):
    """Get basic transforms or augmented versions for TTA"""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Basic transform
    basic_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if not augment:
        return basic_transform

    # Add augmented transforms for TTA
    tta_transforms = [
        basic_transform,  # Original
        # Flip
        transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        # Rotate +5°
        transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation(degrees=(5, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        # Rotate -5°
        transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation(degrees=(-5, -5)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        # Adjust brightness
        transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    ]
    return tta_transforms


def predict(model, image, device, transforms, use_tta=False):
    """Run prediction with optional TTA"""
    model.eval()

    with torch.no_grad():
        if not use_tta:
            # Standard prediction
            tensor = transforms(image).unsqueeze(0).to(device)
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)
            return prediction.item(), confidence.item()
        else:
            # Test-time augmentation
            all_outputs = []
            for transform in transforms:
                tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(tensor)
                all_outputs.append(outputs)

            # Average predictions
            avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
            probs = torch.nn.functional.softmax(avg_outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)
            return prediction.item(), confidence.item()


def evaluate_samples(
    model,
    image_paths,
    class_mapping,
    device,
    output_dir="plots",
    image_size=244,
    use_tta=False,
):
    """Evaluate and visualize model performance on samples"""
    os.makedirs(output_dir, exist_ok=True)

    # Get transforms
    transforms_list = get_transforms(image_size, augment=use_tta)

    results = []
    for img_path in tqdm(image_paths, desc="Evaluating samples"):
        # Extract true class from path
        true_class = Path(img_path).parent.name

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Make prediction
        pred_class, confidence = predict(
            model,
            image,
            device,
            transforms_list if not use_tta else transforms_list,
            use_tta=use_tta,
        )

        # Store result
        results.append(
            {
                "image": img_path,
                "true_class": true_class,
                "predicted_class": class_mapping[pred_class],
                "confidence": confidence,
                "correct": class_mapping[pred_class] == true_class,
            }
        )

    # Calculate accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)

    # Create visualization
    visualize(results[:6], use_tta, class_mapping, output_dir)

    return results, accuracy


def visualize(results, use_tta, class_mapping, output_dir):
    """Create visualization grid of predictions"""
    n_images = len(results)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for i, result in enumerate(results):
        if i >= len(axes):
            break

        # Load and display image
        img = Image.open(result["image"])
        axes[i].imshow(img)

        # Set title
        title = f"True: {result['true_class']}\nPred: {result['predicted_class']} ({result['confidence']:.2f})"
        color = "green" if result["correct"] else "red"
        axes[i].set_title(title, color=color)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if use_tta:
        plt.savefig(os.path.join(output_dir, "tta_comparison.png"), dpi=300)

    else:
        plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=300)
    plt.close()


def get_sample_paths(dataset_path, class_mapping, samples_per_class=2):
    """Get sample image paths for each class"""
    sample_paths = []

    for class_idx, class_name in class_mapping.items():
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue

        # Get first N samples for each class
        image_files = sorted(os.listdir(class_dir))[:samples_per_class]
        sample_paths.extend(
            [os.path.join(class_dir, img) for img in image_files]
        )

    return sample_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained model"
    )
    parser.add_argument("--model-path", default="train_logs/model_best.pth")
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--dataset-path", default="./cleaned_dataset/val")
    parser.add_argument("--train-path", default="./cleaned_dataset/train")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--image-size", type=int, default=244)
    parser.add_argument(
        "--use-tta", action="store_true", help="Use test-time augmentation"
    )
    args = parser.parse_args()

    # Set up
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, device = load_model(
        args.model_path, args.model_name, args.num_classes
    )

    # Get class mapping
    class_mapping = {
        i: d
        for i, d in enumerate(
            sorted(
                [
                    d
                    for d in os.listdir(args.train_path)
                    if os.path.isdir(os.path.join(args.train_path, d))
                ]
            )
        )
    }

    print(f"Model: {args.model_name}, Classes: {len(class_mapping)}")

    # Get samples to evaluate
    sample_paths = get_sample_paths(
        args.dataset_path, class_mapping, args.samples_per_class
    )

    # Run inference
    print(
        f"\nRunning inference on {len(sample_paths)} samples "
        f"{'with' if args.use_tta else 'without'} test-time augmentation..."
    )

    results, accuracy = evaluate_samples(
        model,
        sample_paths,
        class_mapping,
        device,
        args.output_dir,
        args.image_size,
        args.use_tta,
    )

    # Print results
    print("\nResults:")
    print("=" * 70)
    print(f"{'Image':<30} {'True Class':<15} {'Prediction':<15} {'Conf':<8}")
    print("-" * 70)

    for result in results:
        img_name = os.path.basename(result["image"])
        print(
            f"{img_name:<30} {result['true_class']:<15} "
            f"{result['predicted_class']:<15} {result['confidence']:.4f}"
        )

    print("\nSummary:")
    print(
        f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})"
    )
    if args.use_tta:
        print(
            f"Visualization saved to {os.path.join(args.output_dir, 'tta_comparison.png')}"
        )

    else:
        print(
            f"Visualization saved to {os.path.join(args.output_dir, 'predictions.png')}"
        )


if __name__ == "__main__":
    main()
