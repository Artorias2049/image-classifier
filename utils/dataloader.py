import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import autoaugment, transforms


class VehicleDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        for idx, class_folder in enumerate(sorted(os.listdir(image_folder))):
            class_path = os.path.join(image_folder, class_folder)
            if os.path.isdir(class_path):
                self.class_to_idx[class_folder] = idx
                for image_name in os.listdir(class_path):
                    self.image_paths.append(
                        os.path.join(class_path, image_name)
                    )
                    self.labels.append(idx)

        print(
            f"Loaded dataset from {image_folder} with {len(self.image_paths)} images "
            f"across {len(self.class_to_idx)} classes."
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Ensure image is float32
        if isinstance(image, torch.Tensor):
            image = image.float()

        return image, label


class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)

        # Generate mixup weights
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)
        lam = (
            torch.tensor(lam, device=images.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        # Create mixup indices - pair each image with another in batch
        indices = torch.randperm(batch_size, device=images.device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]

        # Return both labels for loss computation
        lam = lam.squeeze()
        return mixed_images, labels, labels[indices], lam


def get_class_weights(dataset):
    class_counts = {}
    for label in dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"Class distribution: {class_counts}")

    # Use effective number of samples instead of inverse frequency
    beta = 0.9999
    effective_num = 1.0 - torch.pow(
        beta, torch.tensor(list(class_counts.values()))
    )
    weights = (1.0 - beta) / effective_num

    # Normalize weights
    weights = weights / weights.sum() * len(class_counts)

    # Create per-sample weights
    sample_weights = [weights[label].item() for label in dataset.labels]
    return sample_weights


def get_dataloaders(
    train_dir,
    val_dir,
    batch_size=32,
    num_workers=4,
    use_mixup=True,
    image_size=244,
    augmentation_strength="strong",
):
    """Create train and validation dataloaders with configurable augmentations."""
    # ImageNet normalization values
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Basic validation transform
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Define augmentation based on strength
    if augmentation_strength == "none":
        train_transform = val_transform
    elif augmentation_strength == "light":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif augmentation_strength == "medium":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.3),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)
                ),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ]
        )

    # Create datasets
    train_dataset = VehicleDataset(train_dir, transform=train_transform)
    val_dataset = VehicleDataset(val_dir, transform=val_transform)

    # Compute class weights for weighted sampling
    sample_weights = get_class_weights(train_dataset)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Create mixup function if enabled
    mixup_fn = Mixup(alpha=0.2) if use_mixup else None

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Augmentation strength: {augmentation_strength}")

    return train_loader, val_loader, mixup_fn
