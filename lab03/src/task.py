import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

CURRENT_FILE = Path(__file__).resolve()
LAB_DIR = next((p for p in [CURRENT_FILE] + list(CURRENT_FILE.parents) if p.name == "lab03"), CURRENT_FILE.parent)
RESULTS_DIR = LAB_DIR / "results"

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_dataset, val_dataset = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class PatchAttention(nn.Module):
    def __init__(self, patch_size=7, in_channels=1, d_model=64, nhead=4, dim_feedforward=128, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, 16, self.patch_dim)
        tokens = self.proj(patches)
        out = self.transformer(tokens)
        pooled = out.mean(dim=1)
        return self.fc(pooled)

def measure_latency(model, batch_size, runs, device):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return float(np.median(latencies))

def train_and_eval(model, train_loader, test_loader, epochs=5, lr=1e-3, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            correct += (preds.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    flops_mlp_linear = 2 * 784 * 256
    flops_cnn_conv2 = 2 * 3 * 3 * 16 * 32 * 14 * 14
    flops_attn_qkv = 2 * 16 * 64 * 3 * 64
    print("--- Hand-calculated Representative Layer FLOPs ---")
    print(f"MLP (Linear 784->256): {flops_mlp_linear:,}")
    print(f"CNN (Conv2d 16->32, 14x14): {flops_cnn_conv2:,}")
    print(f"Attention (QKV Proj, seq=16, d=64): {flops_attn_qkv:,}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    models = {
        "MLP": MLP().to(device),
        "CNN": CNN().to(device),
        "Attention": PatchAttention().to(device)
    }

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model_stats = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        param_count = model_stats.total_params

        dummy_single = torch.randn(1, 1, 28, 28).to(device)
        macs, _ = profile(model, inputs=(dummy_single,), verbose=False)
        measured_flops = macs * 2

        lat_b1 = measure_latency(model, batch_size=1, runs=100, device=device)
        lat_b128 = measure_latency(model, batch_size=128, runs=20, device=device)

        print(f"Training {name} for 5 epochs...")
        acc = train_and_eval(model, train_loader, test_loader, epochs=5, lr=1e-3, device=device)

        results[name] = {
            "params": param_count,
            "flops": measured_flops,
            "lat_b1": lat_b1,
            "lat_b128": lat_b128,
            "acc": acc
        }
        print(f"[{name}] Params: {param_count:,} | THOP FLOPs: {measured_flops:,.0f} | Lat B=1: {lat_b1:.4f} ms | Lat B=128: {lat_b128:.4f} ms | Test Acc: {acc:.4f}")

    names = list(results.keys())
    accs = [results[n]["acc"] * 100 for n in names]
    params = [results[n]["params"] / 1e3 for n in names]
    lat_b128 = [results[n]["lat_b128"] for n in names]

    plt.figure(figsize=(7, 5))
    for i, name in enumerate(names):
        plt.scatter(params[i], accs[i], s=120, label=name)
        plt.annotate(f"{name} ({accs[i]:.2f}%)", (params[i] + 1, accs[i] - 0.2))
    plt.xlabel("Parameters (K)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Parameters")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(RESULTS_DIR / "accuracy_vs_params.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    for i, name in enumerate(names):
        plt.scatter(lat_b128[i], accs[i], s=120, label=name)
        plt.annotate(f"{name} ({accs[i]:.2f}%)", (lat_b128[i] * 1.02, accs[i] - 0.2))
    plt.xlabel("Inference Latency Batch 128 (ms)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Latency (Batch 128)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFinished. Plots saved to {RESULTS_DIR}")