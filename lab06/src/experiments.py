import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import build_loaders
from .engine import evaluate, fit_model, load_checkpoint, set_seed
from .model import build_model


def run_scheduler_and_optimizer_study(
    cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str
) -> dict:
    device = cfg["device"]
    epochs = cfg["benchmarks"]["schedule_epochs"]
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )
    criterion = nn.CrossEntropyLoss()

    schedule_names = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    schedule_results = {}

    for name in schedule_names:
        print(f"   -> testing scheduler={name}...", flush=True)
        set_seed(cfg["seed"])
        model = build_model(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        elif name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=1
            )

        hist = fit_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device, scheduler=scheduler
        )
        _, test_acc = evaluate(model, test_loader, criterion, device=device)
        schedule_results[name] = {
            "final_val_loss": round(hist["val_loss"][-1], 4),
            "final_val_acc": round(hist["val_acc"][-1], 4),
            "test_acc": round(test_acc, 4),
            "history": hist,
        }

    opt_configs = {
        "Adam (lr=1e-3)": lambda params: torch.optim.Adam(params, lr=1e-3),
        "SGD+Momentum (lr=0.05, m=0.9)": lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.9),
    }
    optimizer_results = {}

    for opt_name, opt_fn in opt_configs.items():
        print(f"   -> testing optimizer={opt_name}...", flush=True)
        set_seed(cfg["seed"])
        model = build_model(cfg).to(device)
        optimizer = opt_fn(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        hist = fit_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device, scheduler=scheduler
        )
        _, test_acc = evaluate(model, test_loader, criterion, device=device)
        optimizer_results[opt_name] = {
            "final_val_loss": round(hist["val_loss"][-1], 4),
            "final_val_acc": round(hist["val_acc"][-1], 4),
            "test_acc": round(test_acc, 4),
            "history": hist,
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs_range = list(range(1, epochs + 1))

    for name, res in schedule_results.items():
        ax1.plot(epochs_range, res["history"]["val_acc"], marker="o", markersize=4, label=name)
    ax1.set_title("LR Schedules Comparison (Adam, 10 Epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for opt_name, res in optimizer_results.items():
        ax2.plot(epochs_range, res["history"]["val_acc"], marker="s", markersize=4, label=opt_name)
    ax2.set_title("SGD+Momentum vs Adam (10 Epochs)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "schedules_and_optimizers.png"), dpi=150)
    plt.close(fig)

    return {"schedules": schedule_results, "optimizers": optimizer_results}


def run_gradient_accumulation_study(
    cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str
) -> dict:
    device = cfg["device"]
    b_cfg = cfg["benchmarks"]
    epochs = b_cfg["accum_epochs"]
    base_bs = b_cfg["accum_base_batch"]
    accum_steps = b_cfg["accum_steps"]
    target_bs = b_cfg["accum_target_batch"]
    criterion = nn.CrossEntropyLoss()

    set_seed(cfg["seed"])
    model_native = build_model(cfg).to(device)
    opt_native = torch.optim.Adam(model_native.parameters(), lr=cfg["training"]["lr"])
    train_256, val_256, test_256 = build_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=target_bs, num_workers=0, seed=cfg["seed"]
    )
    hist_native = fit_model(
        model_native, train_256, val_256, criterion, opt_native, epochs=epochs, device=device, accum_steps=1
    )
    _, test_acc_native = evaluate(model_native, test_256, criterion, device=device)

    set_seed(cfg["seed"])
    model_accum = build_model(cfg).to(device)
    opt_accum = torch.optim.Adam(model_accum.parameters(), lr=cfg["training"]["lr"])
    train_64, val_64, test_64 = build_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=base_bs, num_workers=0, seed=cfg["seed"]
    )
    hist_accum = fit_model(
        model_accum,
        train_64,
        val_64,
        criterion,
        opt_accum,
        epochs=epochs,
        device=device,
        accum_steps=accum_steps,
    )
    _, test_acc_accum = evaluate(model_accum, test_64, criterion, device=device)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = list(range(1, epochs + 1))
    ax.plot(ep, hist_native["val_loss"], marker="o", linewidth=2, label=f"Native Batch {target_bs}")
    ax.plot(
        ep,
        hist_accum["val_loss"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Batch {base_bs} x Accum {accum_steps} (Effective {base_bs * accum_steps})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Gradient Accumulation Equivalence Check")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_accumulation.png"), dpi=150)
    plt.close(fig)

    return {
        "native_256": {
            "final_val_loss": round(hist_native["val_loss"][-1], 4),
            "final_val_acc": round(hist_native["val_acc"][-1], 4),
            "test_acc": round(test_acc_native, 4),
            "val_loss_curve": [round(x, 4) for x in hist_native["val_loss"]],
        },
        "accum_64x4": {
            "final_val_loss": round(hist_accum["val_loss"][-1], 4),
            "final_val_acc": round(hist_accum["val_acc"][-1], 4),
            "test_acc": round(test_acc_accum, 4),
            "val_loss_curve": [round(x, 4) for x in hist_accum["val_loss"]],
        },
    }


def run_interrupt_resume_study(
    cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str
) -> dict:
    device = cfg["device"]
    total_epochs = cfg["benchmarks"]["resume_total_epochs"]
    interrupt_epoch = cfg["benchmarks"]["resume_interrupt_epoch"]
    ckpt_path = os.path.join(results_dir, "checkpoints", "interrupt_epoch3.pt")
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        seed=cfg["seed"],
    )

    set_seed(cfg["seed"])
    model_part1 = build_model(cfg).to(device)
    opt_part1 = torch.optim.Adam(model_part1.parameters(), lr=cfg["training"]["lr"])
    sched_part1 = torch.optim.lr_scheduler.StepLR(opt_part1, step_size=2, gamma=0.7)

    fit_model(
        model_part1,
        train_loader,
        val_loader,
        criterion,
        opt_part1,
        epochs=interrupt_epoch,
        device=device,
        scheduler=sched_part1,
        checkpoint_path=ckpt_path,
        start_epoch=1,
    )

    model_resumed = build_model(cfg).to(device)
    opt_resumed = torch.optim.Adam(model_resumed.parameters(), lr=cfg["training"]["lr"])
    sched_resumed = torch.optim.lr_scheduler.StepLR(opt_resumed, step_size=2, gamma=0.7)

    saved_epoch, loaded_history = load_checkpoint(
        ckpt_path, model_resumed, optimizer=opt_resumed, scheduler=sched_resumed, device=device
    )

    part1_loss_snapshot = list(loaded_history["val_loss"])

    full_history = fit_model(
        model_resumed,
        train_loader,
        val_loader,
        criterion,
        opt_resumed,
        epochs=total_epochs,
        device=device,
        scheduler=sched_resumed,
        start_epoch=saved_epoch + 1,
        history=loaded_history,
    )
    _, resumed_test_acc = evaluate(model_resumed, test_loader, criterion, device=device)

    set_seed(cfg["seed"])
    model_ref = build_model(cfg).to(device)
    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=cfg["training"]["lr"])
    sched_ref = torch.optim.lr_scheduler.StepLR(opt_ref, step_size=2, gamma=0.7)
    ref_history = fit_model(
        model_ref,
        train_loader,
        val_loader,
        criterion,
        opt_ref,
        epochs=total_epochs,
        device=device,
        scheduler=sched_ref,
        start_epoch=1,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep_all = list(range(1, total_epochs + 1))
    ep_p1 = list(range(1, interrupt_epoch + 1))
    ep_p2 = list(range(interrupt_epoch, total_epochs + 1))

    ax.plot(ep_all, ref_history["val_loss"], color="gray", linewidth=3, alpha=0.45, label="Uninterrupted Reference")
    ax.plot(ep_p1, part1_loss_snapshot, marker="o", color="tab:blue", linewidth=2, label="Phase 1 (Epochs 1–3)")
    ax.plot(
        ep_p2,
        full_history["val_loss"][interrupt_epoch - 1 :],
        marker="s",
        linestyle="--",
        color="tab:orange",
        linewidth=2,
        label="Resumed Phase 2 (Epochs 3–6)",
    )
    ax.axvline(x=interrupt_epoch, color="tab:red", linestyle=":", label="Interrupt & Checkpoint")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Checkpoint Interrupt & Smooth Resume Verification")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "interrupt_resume.png"), dpi=150)
    plt.close(fig)

    return {
        "interrupted_at_epoch": saved_epoch,
        "resumed_val_loss_curve": [round(x, 4) for x in full_history["val_loss"]],
        "reference_val_loss_curve": [round(x, 4) for x in ref_history["val_loss"]],
        "resumed_test_acc": round(resumed_test_acc, 4),
    }