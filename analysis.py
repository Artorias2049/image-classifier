import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def get_image_sizes(image_folder):
    widths, heights, aspect_ratios = [], [], []
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return widths, heights, aspect_ratios


def get_class_distribution(image_folder):
    class_counts = {}
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
    return class_counts


def compute_rgb_statistics(image_folder):
    means, stds = [], []
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)
    return mean_rgb, std_rgb


def plot_class_distribution(class_counts, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45)
    plt.xlabel("Vehicle Type")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()


def plot_image_size_distribution(widths, heights, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(widths, bins=30, kde=True, color="blue", label="Width")
    sns.histplot(heights, bins=30, kde=True, color="red", label="Height")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("Image Size Distribution")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(
        os.path.join(output_dir, "image_size_distribution.png"), dpi=300
    )
    plt.close()


def plot_aspect_ratio_distribution(aspect_ratios, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(aspect_ratios, bins=30, kde=True, color="purple")
    plt.xlabel("Aspect Ratio (Width / Height)")
    plt.ylabel("Frequency")
    plt.title("Aspect Ratio Distribution")
    plt.tight_layout()
    # Save the plot
    plt.savefig(
        os.path.join(output_dir, "aspect_ratio_distribution.png"), dpi=300
    )
    plt.close()


def plot_imbalance_pie_chart(class_counts, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.pie(
        class_counts.values(),
        labels=class_counts.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2"),
    )
    plt.title("Dataset Imbalance (Class Distribution)")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "class_imbalance_pie.png"), dpi=300)
    plt.close()


def plot_outlier_detection(widths, heights, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=widths, color="blue")
    plt.title("Outlier Detection - Image Widths")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "width_outliers.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=heights, color="red")
    plt.title("Outlier Detection - Image Heights")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, "height_outliers.png"), dpi=300)
    plt.close()


def plot_rgb_distribution(mean_rgb, std_rgb, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    channels = ["Red", "Green", "Blue"]

    # Mean RGB values
    plt.figure(figsize=(8, 6))
    plt.bar(channels, mean_rgb, color=["red", "green", "blue"])
    plt.title("Mean RGB Values Across Dataset")
    plt.ylabel("Pixel Value (0-255)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rgb_means.png"), dpi=300)
    plt.close()

    # Standard deviation of RGB values
    plt.figure(figsize=(8, 6))
    plt.bar(channels, std_rgb, color=["darkred", "darkgreen", "darkblue"])
    plt.title("Standard Deviation of RGB Values")
    plt.ylabel("Standard Deviation")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rgb_stds.png"), dpi=300)
    plt.close()


def compare_class_distributions(train_counts, val_counts, output_dir="plots"):
    """Compare class distributions between train and validation sets"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique classes
    all_classes = sorted(
        set(list(train_counts.keys()) + list(val_counts.keys()))
    )

    # Create lists for plotting
    train_values = [train_counts.get(cls, 0) for cls in all_classes]
    val_values = [val_counts.get(cls, 0) for cls in all_classes]

    # Calculate ratios for percentage comparison
    train_percentages = [
        count / sum(train_values) * 100 for count in train_values
    ]
    val_percentages = [count / sum(val_values) * 100 for count in val_values]

    # Set up the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart with absolute counts
    x = np.arange(len(all_classes))
    width = 0.35
    ax1.bar(
        x - width / 2,
        train_values,
        width,
        label="Train",
        color="blue",
        alpha=0.7,
    )
    ax1.bar(
        x + width / 2,
        val_values,
        width,
        label="Validation",
        color="orange",
        alpha=0.7,
    )

    # Configure the first y-axis (absolute counts)
    ax1.set_ylabel("Number of Images", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_classes, rotation=45, ha="right")
    ax1.legend(loc="upper left")

    # Add second y-axis for percentage comparison
    ax2 = ax1.twinx()
    ax2.plot(x, train_percentages, "b-", marker="o", label="Train %")
    ax2.plot(
        x, val_percentages, "orange", marker="s", linestyle="--", label="Val %"
    )
    ax2.set_ylabel("Percentage of Dataset (%)", fontsize=12)
    ax2.legend(loc="upper right")

    # Add title and adjust layout
    plt.title("Train vs Validation Class Distribution", fontsize=14)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, "train_val_comparison.png"), dpi=300)
    plt.close()

    # Create a stacked bar chart for distribution visualization
    plt.figure(figsize=(14, 7))

    # Calculate the percentages for each dataset
    train_pct = {
        cls: 100 * train_counts.get(cls, 0) / sum(train_counts.values())
        for cls in all_classes
    }
    val_pct = {
        cls: 100 * val_counts.get(cls, 0) / sum(val_counts.values())
        for cls in all_classes
    }

    # Sort classes by train percentage
    sorted_classes = sorted(
        all_classes, key=lambda cls: train_pct[cls], reverse=True
    )
    train_pct_sorted = [train_pct[cls] for cls in sorted_classes]
    val_pct_sorted = [val_pct[cls] for cls in sorted_classes]

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot horizontal bars
    y_pos = np.arange(len(sorted_classes))
    plt.barh(
        y_pos,
        train_pct_sorted,
        height=0.4,
        label="Train",
        color="blue",
        alpha=0.7,
    )
    plt.barh(
        y_pos + 0.4,
        val_pct_sorted,
        height=0.4,
        label="Validation",
        color="orange",
        alpha=0.7,
    )

    # Add labels and title
    plt.yticks(y_pos + 0.2, sorted_classes)
    plt.xlabel("Percentage of Dataset (%)")
    plt.title("Train vs Validation Class Distribution (Percentage)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "train_val_distribution_pct.png"), dpi=300
    )
    plt.close()


def analyze_dataset_split(split_name, image_folder, output_dir):
    """Analyze a single dataset split (train or validation)"""
    print(f"Analyzing {split_name} dataset...")
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Get dataset statistics
    class_counts = get_class_distribution(image_folder)
    widths, heights, aspect_ratios = get_image_sizes(image_folder)
    mean_rgb, std_rgb = compute_rgb_statistics(image_folder)

    print(f"{split_name} Class Counts: {class_counts}")
    print(f"{split_name} Mean RGB: {mean_rgb}")
    print(f"{split_name} Standard Deviation RGB: {std_rgb}")

    # Save analysis statistics as text
    with open(os.path.join(split_dir, f"{split_name}_analysis.txt"), "w") as f:
        f.write(f"=== {split_name.capitalize()} Dataset Analysis ===\n\n")
        f.write("Class Distribution:\n")
        for cls, count in class_counts.items():
            f.write(f"{cls}: {count} images\n")

        f.write(f"\nTotal classes: {len(class_counts)}\n")
        f.write(f"Total images: {sum(class_counts.values())}\n")

        f.write("\nImage Size Statistics:\n")
        f.write(f"Total images analyzed: {len(widths)}\n")
        f.write(
            f"Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}, Median: {np.median(widths)}\n"
        )
        f.write(
            f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}, Median: {np.median(heights)}\n"
        )

        f.write("\nAspect Ratio Statistics:\n")
        f.write(
            f"Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}\n"
        )

        f.write("\nRGB Statistics:\n")
        f.write(
            f"Mean RGB: [{mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f}]\n"
        )
        f.write(
            f"Std RGB: [{std_rgb[0]:.2f}, {std_rgb[1]:.2f}, {std_rgb[2]:.2f}]\n"
        )

    # Generate plots for this split
    plot_class_distribution(class_counts, split_dir)
    plot_image_size_distribution(widths, heights, split_dir)
    plot_aspect_ratio_distribution(aspect_ratios, split_dir)
    plot_imbalance_pie_chart(class_counts, split_dir)
    plot_outlier_detection(widths, heights, split_dir)
    plot_rgb_distribution(mean_rgb, std_rgb, split_dir)

    return class_counts, (widths, heights, aspect_ratios), (mean_rgb, std_rgb)


def main():
    train_folder = "vehicle_dataset/train"
    val_folder = "vehicle_dataset/val"
    output_dir = "plots/analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze train and validation datasets separately
    train_class_counts, train_size_stats, train_rgb_stats = (
        analyze_dataset_split("train", train_folder, output_dir)
    )
    val_class_counts, val_size_stats, val_rgb_stats = analyze_dataset_split(
        "val", val_folder, output_dir
    )

    # Compare train and validation distributions
    compare_class_distributions(
        train_class_counts, val_class_counts, output_dir
    )

    # Compute dataset-wide statistics
    train_widths, train_heights, train_aspect_ratios = train_size_stats
    val_widths, val_heights, val_aspect_ratios = val_size_stats

    # Combine train and validation statistics for overall analysis
    all_widths = train_widths + val_widths
    all_heights = train_heights + val_heights
    all_aspect_ratios = train_aspect_ratios + val_aspect_ratios

    # Create combined analysis plots
    plot_image_size_distribution(all_widths, all_heights, output_dir)
    plot_aspect_ratio_distribution(all_aspect_ratios, output_dir)

    # Calculate stats for entire dataset
    all_counts = {
        k: train_class_counts.get(k, 0) + val_class_counts.get(k, 0)
        for k in set(train_class_counts) | set(val_class_counts)
    }

    # Save combined dataset statistics
    with open(
        os.path.join(output_dir, "combined_dataset_analysis.txt"), "w"
    ) as f:
        f.write("=== Combined Dataset Analysis ===\n\n")
        f.write("Class Distribution:\n")
        for cls, count in all_counts.items():
            train_count = train_class_counts.get(cls, 0)
            val_count = val_class_counts.get(cls, 0)
            train_pct = 100 * train_count / count if count > 0 else 0
            val_pct = 100 * val_count / count if count > 0 else 0
            f.write(
                f"{cls}: {count} images (Train: {train_count} [{train_pct:.1f}%], Val: {val_count} [{val_pct:.1f}%])\n"
            )

        f.write(f"\nTotal classes: {len(all_counts)}\n")
        f.write(f"Total images: {sum(all_counts.values())}\n")
        f.write(
            f"Train/Val split: {sum(train_class_counts.values())}/{sum(val_class_counts.values())} "
        )
        f.write(
            f"({100*sum(train_class_counts.values())/sum(all_counts.values()):.1f}%/"
        )
        f.write(
            f"{100*sum(val_class_counts.values())/sum(all_counts.values()):.1f}%)\n"
        )

        f.write("\nImage Size Statistics (Combined):\n")
        f.write(f"Total images analyzed: {len(all_widths)}\n")
        f.write(
            f"Width - Min: {min(all_widths)}, Max: {max(all_widths)}, Mean: {np.mean(all_widths):.2f}\n"
        )
        f.write(
            f"Height - Min: {min(all_heights)}, Max: {max(all_heights)}, Mean: {np.mean(all_heights):.2f}\n"
        )

        f.write("\nAspect Ratio Statistics (Combined):\n")
        f.write(
            f"Min: {min(all_aspect_ratios):.2f}, Max: {max(all_aspect_ratios):.2f}, Mean: {np.mean(all_aspect_ratios):.2f}\n"
        )

    # Generate overall dataset visualizations
    plot_imbalance_pie_chart(all_counts, output_dir)

    print(f"All analysis plots saved to {output_dir}")


if __name__ == "__main__":
    main()
