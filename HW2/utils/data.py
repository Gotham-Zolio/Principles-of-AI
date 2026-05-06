import os
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 96, augment: bool = False):
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def create_dataloaders(
    train_root: str,
    test_root: str,
    image_size: int = 96,
    valid_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    augment: bool = False,
) -> Tuple[Dict[str, DataLoader], List[str]]:
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    train_tf, eval_tf = build_transforms(image_size=image_size, augment=augment)

    full_for_split = datasets.ImageFolder(train_root)
    targets = [label for _, label in full_for_split.samples]
    indices = list(range(len(targets)))

    train_indices, valid_indices = train_test_split(
        indices,
        test_size=valid_ratio,
        random_state=seed,
        stratify=targets,
    )

    train_dataset = datasets.ImageFolder(train_root, transform=train_tf)
    valid_dataset = datasets.ImageFolder(train_root, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_tf)

    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)

    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            valid_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders, full_for_split.classes
