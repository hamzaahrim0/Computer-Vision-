"""Transforms, Dataset and DataLoaders for UC Merced (Phase 1)."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from .splits import build_index, get_split

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRotate90:
    """Rotate the image by a random multiple of 90 degrees."""

    def __call__(self, img):
        k = random.randint(0, 3)
        return TF.rotate(img, 90 * k) if k else img


def get_transforms(train, img_size=224):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # mild crop
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRotate90(),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])


class UCMercedDataset(Dataset):
    def __init__(self, root, samples, indices, transform):
        self.root = Path(root)
        self.samples = [samples[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel_path, label = self.samples[i]
        img = Image.open(self.root / rel_path).convert("RGB")
        return self.transform(img), label


def _worker_init(worker_id):
    # Different, reproducible seed for each worker (affects `random`)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)


def get_dataloaders(cfg, seed):
    """Build train/test DataLoaders for a given seed.

    Returns: train_loader, test_loader, classes
    """
    root = cfg["data"]["root"]
    samples, classes = build_index(root)
    train_idx, test_idx = get_split(
        samples, classes, cfg["data"]["test_size"], seed,
        cfg["paths"]["results_dir"],
    )

    size = cfg["data"]["img_size"]
    train_ds = UCMercedDataset(root, samples, train_idx, get_transforms(True, size))
    test_ds = UCMercedDataset(root, samples, test_idx, get_transforms(False, size))

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=g,
                              drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, test_loader, classes
