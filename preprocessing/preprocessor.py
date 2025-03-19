import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


class ImagePreprocessor:
    """Class to handle image preprocessing with configurable parameters"""

    def __init__(self, target_size=(180, 180), padding_color=(255, 255, 255)):
        self.target_size = target_size
        self.padding_color = padding_color

    def resize_with_padding(self, image, interpolation=cv2.INTER_AREA):
        """Resize an image while maintaining aspect ratio with padding."""
        h, w = image.shape[:2]

        # Compute scaling factor while preserving aspect ratio
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Create a blank canvas with padding color
        padded_img = np.full(
            (self.target_size[0], self.target_size[1], 3),
            self.padding_color,
            dtype=np.uint8,
        )

        # Compute padding offsets to center the image
        x_offset = (self.target_size[1] - new_w) // 2
        y_offset = (self.target_size[0] - new_h) // 2

        # Place resized image on the center of the canvas
        padded_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            resized
        )

        return padded_img

    def process_image(self, input_path, output_path):
        """Process a single image."""
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"Warning: Unable to read {input_path}. Skipping.")
                return False

            padded_img = self.resize_with_padding(img)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, padded_img)
            return True

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False


def process_single_image(args):
    """Helper function for multiprocessing"""
    input_path, output_path, target_size, padding_color = args
    preprocessor = ImagePreprocessor(target_size, padding_color)
    return preprocessor.process_image(input_path, output_path)


def clean_problematic_directories(dir_path):
    """Remove any directories with image file extensions"""
    if not os.path.exists(dir_path):
        return

    print(f"Cleaning problematic directories in {dir_path}")
    removed = 0

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for d in dirs:
            if d.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                try:
                    path = os.path.join(root, d)
                    print(f"Removing problematic directory: {path}")
                    shutil.rmtree(path)
                    removed += 1
                except Exception as e:
                    print(f"Error removing {os.path.join(root, d)}: {e}")

    print(f"Removed {removed} problematic directories")


def preprocess_dataset(
    dataset_path,
    output_path,
    target_size=(180, 180),
    padding_color=(255, 255, 255),
    num_workers=None,
):
    """Preprocess all images in a dataset."""
    if os.path.exists(output_path):
        # First try to clean any problematic directories
        clean_problematic_directories(output_path)

    # Ensure the base directories exist
    train_output = os.path.join(output_path, "train")
    val_output = os.path.join(output_path, "val")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, num_workers)

    # Manually create the train/val splits
    print(f"Dataset path: {dataset_path}")

    tasks = []

    # Process train split
    train_path = os.path.join(dataset_path, "train")
    if os.path.exists(train_path):
        print(f"Processing train split from {train_path}")
        train_classes = 0
        train_images = 0

        # Process each class
        for class_name in os.listdir(train_path):
            class_dir = os.path.join(train_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Create output class directory
            train_class_output = os.path.join(train_output, class_name)
            os.makedirs(train_class_output, exist_ok=True)
            train_classes += 1

            # Process each image
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    continue

                output_img_path = os.path.join(train_class_output, img_name)

                # Skip if already processed
                if os.path.exists(output_img_path):
                    continue

                tasks.append(
                    (img_path, output_img_path, target_size, padding_color)
                )
                train_images += 1

        print(
            f"Found {train_classes} classes and {train_images} images in train split"
        )
    else:
        print(f"Warning: Train directory {train_path} not found")

    # Process val split
    val_path = os.path.join(dataset_path, "val")
    if os.path.exists(val_path):
        print(f"Processing val split from {val_path}")
        val_classes = 0
        val_images = 0

        # Process each class
        for class_name in os.listdir(val_path):
            class_dir = os.path.join(val_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Create output class directory
            val_class_output = os.path.join(val_output, class_name)
            os.makedirs(val_class_output, exist_ok=True)
            val_classes += 1

            # Process each image
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    continue

                output_img_path = os.path.join(val_class_output, img_name)

                # Skip if already processed
                if os.path.exists(output_img_path):
                    continue

                tasks.append(
                    (img_path, output_img_path, target_size, padding_color)
                )
                val_images += 1

        print(
            f"Found {val_classes} classes and {val_images} images in val split"
        )
    else:
        print(f"Warning: Val directory {val_path} not found")

    # Process all collected tasks
    total_images = len(tasks)
    print(f"Preprocessing {total_images} images using {num_workers} workers...")

    if total_images == 0:
        print("No images found to process")
        return

    processed_images = 0
    failed_images = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total_images, desc="Processing images") as pbar:
            futures = {
                executor.submit(process_single_image, task): task
                for task in tasks
            }

            for future in as_completed(futures):
                if future.result():
                    processed_images += 1
                else:
                    failed_images += 1
                pbar.update(1)

    print("Dataset preprocessing complete.")
    print(f"Processed {processed_images} images successfully.")
    if failed_images > 0:
        print(f"Failed to process {failed_images} images.")

    # Verify the structure
    print("Output directory structure:")
    print(f"- {output_path}")
    for split in ["train", "val"]:
        split_path = os.path.join(output_path, split)
        if os.path.exists(split_path):
            class_count = len(
                [
                    d
                    for d in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, d))
                ]
            )
            print(f"  - {split}: {class_count} classes")
