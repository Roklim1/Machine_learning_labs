import argparse
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml

from .data import build_loaders, get_datasets
from .engine import evaluate, fit_model, set_seed
from .experiments import (
    run_gradient_accumulation_study,
    run_interrupt_resume_study,
    run_scheduler_and_optimizer_study,
)
from .model import build_model
from .profiler_bench import (
    benchmark_batch_sizes,
    benchmark_num_workers,
    run_profiler_epoch,
)

LAB_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (LAB_ROOT / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="Lab 06: Training Pipelines, Profiling, and Optimization")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["root"] = str(resolve_path(cfg["data"]["root"]))
    results_dir = resolve_path(cfg["output"]["results_dir"])
    checkpoint_dir = resolve_path(cfg["training"]["checkpoint_dir"])
    mlruns_dir = resolve_path(cfg["mlflow"]["tracking_uri"])

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = cfg["device"]

    train_dataset, val_dataset, test_dataset = get_datasets(cfg)
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )

    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    summary = {}

    print("[1/7] Running baseline YAML-driven training with MLflow & checkpointing...", flush=True)
    with mlflow.start_run(run_name="baseline_lab3_cnn"):
        mlflow.log_params(
            {
                "seed": cfg["seed"],
                "device": device,
                "batch_size": cfg["data"]["batch_size"],
                "num_workers": cfg["data"]["num_workers"],
                "epochs": cfg["training"]["epochs"],
                "lr": cfg["training"]["lr"],
                "optimizer": cfg["training"]["optimizer"],
                "conv1_channels": cfg["model"]["conv1_channels"],
                "conv2_channels": cfg["model"]["conv2_channels"],
            }
        )

        model = build_model(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_params", total_params)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
        ckpt_path = str(checkpoint_dir / "baseline_best.pt")

        history = fit_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=cfg["training"]["epochs"],
            device=device,
            checkpoint_path=ckpt_path,
            mlflow_module=mlflow,
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(ckpt_path)

        summary["task1_baseline"] = {
            "total_params": total_params,
            "final_val_acc": round(history["val_acc"][-1], 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
        }

    print("[2/7] Profiling 1 epoch with torch.profiler...", flush=True)
    summary["task2_profiler"] = run_profiler_epoch(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    print("[3/7] Benchmarking DataLoader num_workers [0, 1, 2, 4]...", flush=True)
    summary["task3_num_workers"] = benchmark_num_workers(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    print("[4/7] Running batch-size study [32, 64, 128, 256, 512]...", flush=True)
    summary["task4_batch_sizes"] = benchmark_batch_sizes(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    print("[5/7] Comparing LR schedules and SGD+Momentum vs Adam (10 epochs)...", flush=True)
    summary["task5_schedules_optimizers"] = run_scheduler_and_optimizer_study(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    print("[6/7] Running gradient accumulation equivalence check (64x4 vs 256)...", flush=True)
    summary["task6_gradient_accumulation"] = run_gradient_accumulation_study(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    print("[7/7] Testing checkpoint interrupt at epoch 3 and resume...", flush=True)
    summary["task7_interrupt_resume"] = run_interrupt_resume_study(
        cfg, train_dataset, val_dataset, test_dataset, str(results_dir)
    )

    summary_path = results_dir / "lab06_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with mlflow.start_run(run_name="lab06_all_artifacts"):
        mlflow.log_artifacts(str(results_dir))

    print(f"Done! Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()