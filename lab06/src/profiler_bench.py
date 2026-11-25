import gc
import os
import time
import tracemalloc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from .data import build_loaders
from .engine import evaluate, fit_model, set_seed
from .model import build_model


def run_profiler_epoch(cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str) -> dict:
    device = cfg["device"]
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    train_loader, _, _ = build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    stage_times = {"data_loading": 0.0, "forward_pass": 0.0, "backward_pass": 0.0, "optimizer_step": 0.0}

    model.train()
    with profile(activities=activities) as prof:
        data_iter = iter(train_loader)
        while True:
            t0 = time.perf_counter()
            with record_function("stage_data_loading"):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    break
            stage_times["data_loading"] += time.perf_counter() - t0

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            t1 = time.perf_counter()
            with record_function("stage_forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            stage_times["forward_pass"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            with record_function("stage_backward_pass"):
                loss.backward()
            stage_times["backward_pass"] += time.perf_counter() - t2

            t3 = time.perf_counter()
            with record_function("stage_optimizer_step"):
                optimizer.step()
            stage_times["optimizer_step"] += time.perf_counter() - t3

    total_stage_time = sum(stage_times.values())
    shares = {k: (v / total_stage_time) * 100.0 for k, v in stage_times.items()}

    table_str = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
    prof_path = os.path.join(results_dir, "profiler_summary.txt")
    with open(prof_path, "w", encoding="utf-8") as f:
        f.write("=== Stage Breakdown (1 Epoch) ===\n")
        for k in stage_times:
            f.write(f"{k}: {stage_times[k]:.4f} s ({shares[k]:.2f}%)\n")
        f.write("\n=== torch.profiler Top Operators ===\n")
        f.write(table_str)

    return {
        "stage_times_s": stage_times,
        "stage_shares_pct": shares,
        "profiler_table_path": prof_path,
    }


def benchmark_num_workers(cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str) -> list:
    device = cfg["device"]
    workers_list = cfg["benchmarks"]["num_workers_list"]
    batch_size = cfg["data"]["batch_size"]
    results = []

    for nw in workers_list:
        print(f"   -> testing num_workers={nw}...", flush=True)
        set_seed(cfg["seed"])
        model = build_model(cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=batch_size, num_workers=nw, seed=cfg["seed"]
        )

        epoch_times = []
        data_wait_times = []
        for _ in range(2):
            model.train()
            t_epoch_start = time.perf_counter()
            wait_acc = 0.0
            t_wait = time.perf_counter()
            for inputs, targets in train_loader:
                wait_acc += time.perf_counter() - t_wait
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                t_wait = time.perf_counter()
            epoch_times.append(time.perf_counter() - t_epoch_start)
            data_wait_times.append(wait_acc)

        del train_loader, val_loader, test_loader
        gc.collect()

        steady_time = epoch_times[1]
        steady_wait = data_wait_times[1]
        throughput = len(train_dataset) / steady_time

        results.append(
            {
                "num_workers": nw,
                "first_epoch_s": round(epoch_times[0], 4),
                "steady_epoch_s": round(steady_time, 4),
                "data_wait_s": round(steady_wait, 4),
                "compute_s": round(steady_time - steady_wait, 4),
                "throughput_samples_s": round(throughput, 1),
            }
        )

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    x_labels = [str(r["num_workers"]) for r in results]
    steady_vals = [r["steady_epoch_s"] for r in results]
    first_vals = [r["first_epoch_s"] for r in results]
    wait_vals = [r["data_wait_s"] for r in results]

    ax1.plot(x_labels, first_vals, marker="s", linestyle="--", label="Epoch 1 Time (incl. spawn)")
    ax1.plot(x_labels, steady_vals, marker="o", linewidth=2, label="Epoch 2 Steady Time")
    ax1.plot(x_labels, wait_vals, marker="^", label="Data Loading Wait Time")
    ax1.set_xlabel("num_workers")
    ax1.set_ylabel("Time per Epoch (seconds)")
    ax1.set_title("DataLoader num_workers Scaling on Fashion-MNIST")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "num_workers_benchmark.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return results


def benchmark_batch_sizes(cfg: dict, train_dataset, val_dataset, test_dataset, results_dir: str) -> list:
    device = cfg["device"]
    batch_sizes = cfg["benchmarks"]["batch_sizes_list"]
    epochs = cfg["benchmarks"]["batch_study_epochs"]
    results = []

    for bs in batch_sizes:
        print(f"   -> testing batch_size={bs}...", flush=True)
        set_seed(cfg["seed"])
        model = build_model(cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=bs,
            num_workers=cfg["data"]["num_workers"],
            seed=cfg["seed"],
        )

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        else:
            tracemalloc.start()

        t_start = time.perf_counter()
        fit_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device)
        total_time = time.perf_counter() - t_start

        if device == "cuda" and torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mem_mb = peak_bytes / (1024**2)

        _, test_acc = evaluate(model, test_loader, criterion, device=device)
        throughput = (len(train_dataset) * epochs) / total_time

        results.append(
            {
                "batch_size": bs,
                "throughput_samples_s": round(throughput, 1),
                "peak_memory_mb": round(peak_mem_mb, 3),
                "final_test_acc": round(test_acc, 4),
                "total_time_s": round(total_time, 3),
            }
        )

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    bs_labels = [str(r["batch_size"]) for r in results]
    throughputs = [r["throughput_samples_s"] for r in results]
    accs = [r["final_test_acc"] * 100.0 for r in results]

    color1 = "tab:blue"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples/s)", color=color1)
    ax1.bar(bs_labels, throughputs, color=color1, alpha=0.65, width=0.5, label="Throughput")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Test Accuracy (%)", color=color2)
    ax2.plot(bs_labels, accs, color=color2, marker="o", linewidth=2, label="Test Accuracy")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"Batch Size Study ({epochs} Epochs): Throughput vs Test Accuracy")
    fig.tight_layout()
    plot_path = os.path.join(results_dir, "batch_size_study.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return results