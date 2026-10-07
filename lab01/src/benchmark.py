import os
import random
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process()
metrics = []
accuracies = {}

os.makedirs("lab01/results", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = round(acc, 4)

    model.fit(X_train, y_train)
    train_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    train_time_median = np.median(train_times)

    sample = X_test[0:1]
    inference_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        inference_times.append((time.perf_counter() - t0) * 1000)  # в мс
    latency_median_ms = np.median(inference_times)

    model_path = f"lab01/results/{name}_model.joblib"
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    mem_mb = process.memory_info().rss / (1024 * 1024)

    metrics.append({
        "Model": name,
        "Accuracy": accuracies[name],
        "Train Time (s)": round(train_time_median, 5),
        "Inference Latency (ms)": round(latency_median_ms, 4),
        "Size (Bytes)": size_bytes,
        "Size (KB)": round(size_kb, 2),
        "Peak Memory (MB)": round(mem_mb, 2)
    })

df_acc = pd.DataFrame(list(accuracies.items()), columns=["Model", "Test Accuracy"])
df_acc.to_csv("lab01/results/baseline_accuracy.csv", index=False)

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("lab01/results/system_metrics.csv", index=False)

print("=== BASELINE ACCURACY ===")
print(df_acc.to_string(index=False))
print("\n=== SYSTEM METRICS ===")
print(df_metrics.to_string(index=False))