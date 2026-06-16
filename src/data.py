from __future__ import annotations

import os
from PIL import Image
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

corruption_types = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", 
    "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    cifar_class: type
    c_dir: str


def dataset_spec(name):
    name = name.lower()
    if name in {"cifar10", "cifar-10"}:
        return DatasetSpec("cifar10", 10, datasets.CIFAR10, "CIFAR-10-C")
    if name in {"cifar100", "cifar-100"}:
        return DatasetSpec("cifar100", 100, datasets.CIFAR100, "CIFAR-100-C")
    raise ValueError(f"Unknown dataset: {name}")


def image_transform(train, image_size = 224):
    ops = []
    if train:
        ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
    ops.extend([
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return transforms.Compose(ops)


def CIFARDataset(name, root, train, image_size = 224, download = True):
    spec = dataset_spec(name)
    return spec.cifar_class(root=root, train=train, download=download, transform=image_transform(train, image_size))


class CIFARCDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        corruption,
        severity,
        image_size = 224,
        transform = None,
    ):
        spec = dataset_spec(dataset)
        base = os.path.join(root, spec.c_dir)
        x_path = os.path.join(base, f"{corruption}.npy")
        y_path = os.path.join(base, "labels.npy")
        arr = np.load(x_path, mmap_mode="r")
        labels = np.load(y_path, mmap_mode="r")
        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = arr[start:end]
        self.labels = labels[start:end]
        self.transform = transform or image_transform(train=False, image_size=image_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(np.asarray(self.images[idx]).astype(np.uint8))
        label = int(self.labels[idx])
        return self.transform(img), label
