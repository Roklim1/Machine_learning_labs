import os
import random
import time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    history: dict = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = "cpu",
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint.get("history", {})


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    accum_steps: int = 1,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    num_batches = len(loader)

    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        raw_loss = criterion(outputs, targets)

        loss = raw_loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()

        batch_size = targets.size(0)
        running_loss += raw_loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str = "cpu",
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def fit_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cpu",
    scheduler=None,
    accum_steps: int = 1,
    checkpoint_path: str = None,
    start_epoch: int = 1,
    history: dict = None,
    mlflow_module=None,
):
    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device=device, accum_steps=accum_steps
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)
        elapsed = time.perf_counter() - t0

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        if mlflow_module is not None:
            mlflow_module.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_s": elapsed,
                },
                step=epoch,
            )

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, history)

    return history