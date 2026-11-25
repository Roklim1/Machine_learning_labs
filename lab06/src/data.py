import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_datasets(cfg: dict):
    d_cfg = cfg["data"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((d_cfg["mean"],), (d_cfg["std"],)),
        ]
    )

    full_train = datasets.FashionMNIST(
        root=d_cfg["root"], train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=d_cfg["root"], train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(cfg["seed"])
    train_dataset, val_dataset = random_split(
        full_train,
        [d_cfg["train_size"], d_cfg["val_size"]],
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset


def build_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
):
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader