import argparse
import os

import torch

from models.models import VehicleClassifier


def export_to_onnx(
    model_path,
    output_path,
    model_name="efficientnet_b0",
    num_classes=12,
    input_shape=(1, 3, 244, 244),
    verify=True,
):
    """Export PyTorch model to ONNX format"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting model to ONNX using device: {device}")

    # Load the model
    model = VehicleClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_layers=False,
        dropout_rate=0.0,  # Set to 0 for inference
        model_name=model_name,
    ).to(device)

    # Load the weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {model_path}")

    # Add softmax layer to match the verification script
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(input_shape, device=device)

    # Export to ONNX
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=True,
    )

    print(f"Model exported successfully to {output_path}")
    print("Input shape:", input_shape)
    print("Output shape:", (1, num_classes))
    print(f"Model architecture: {model_name}")

    # Verify the export if requested
    if verify:
        try:
            import onnx
            import onnxruntime as ort

            # Check model structure
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure verification successful!")

            # Basic inference test
            session = ort.InferenceSession(output_path)
            input_name = session.get_inputs()[0].name
            test_input = dummy_input.cpu().numpy()
            outputs = session.run(None, {input_name: test_input})

            print("ONNX runtime verification successful!")
            print(f"Output shape: {outputs[0].shape}")
        except ImportError as e:
            print(f"ONNX verification packages not installed: {e}")
            print(
                "To verify, install onnx and onnxruntime: pip install onnx onnxruntime"
            )
        except Exception as e:
            print(f"ONNX model verification failed: {e}")


def generate_classes_txt(dataset_path, output_path="classes.txt"):
    """Generate classes.txt file with class names in order"""
    # Make sure the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist")

    # Get sorted list of class directories
    class_dirs = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )

    if not class_dirs:
        raise ValueError(f"No class directories found in {dataset_path}")

    # Create the classes.txt file
    with open(output_path, "w") as f:
        for i, class_name in enumerate(class_dirs):
            f.write(f"{class_name}")
            if i < len(class_dirs) - 1:
                f.write("\n")

    print(f"Classes written to {output_path}")
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")

    return class_dirs


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_logs/model_best.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="vehicle_classifier.onnx",
        help="Path to save the ONNX model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b0",
        help="Model architecture name",
    )
    parser.add_argument(
        "--num-classes", type=int, default=12, help="Number of output classes"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./cleaned_dataset/train",
        help="Path to the training dataset for class names",
    )
    parser.add_argument(
        "--classes-output",
        type=str,
        default="classes.txt",
        help="Path to save the classes.txt file",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip ONNX model verification"
    )

    args = parser.parse_args()

    # Export the model to ONNX
    export_to_onnx(
        args.model_path,
        args.output_path,
        args.model_name,
        args.num_classes,
        verify=not args.no_verify,
    )

    # Generate classes.txt file
    generate_classes_txt(args.dataset_path, args.classes_output)

    print(f"Export completed successfully. Model saved to {args.output_path}")
    print(f"Class names saved to {args.classes_output}")


if __name__ == "__main__":
    main()
