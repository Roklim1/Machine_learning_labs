import os
import math
import time
import json
import random
import statistics
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = max(min(self.data, 50.0), -50.0)
        val = math.exp(x)
        out = Value(val, (self,), "exp")

        def _backward():
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, 1e-12)
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        if self.data >= 0:
            z = math.exp(-self.data)
            s = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            s = z / (1.0 + z)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class ValueArray:
    def __init__(self, items):
        self.items = list(items)

    def __add__(self, other):
        return ValueArray([a + b for a, b in zip(self.items, other.items)])

    def __mul__(self, other):
        return ValueArray([a * b for a, b in zip(self.items, other.items)])

    def __pow__(self, p):
        return ValueArray([a ** p for a in self.items])

    def relu(self):
        return ValueArray([a.relu() for a in self.items])

    def sum(self):
        total = Value(0.0)
        for item in self.items:
            total = total + item
        return total


def verify_gradients():
    rng = np.random.default_rng(42)
    a_np = rng.normal(0.5, 1.0, size=5)
    b_np = rng.normal(-0.3, 1.0, size=5)
    c_np = rng.normal(0.2, 1.0, size=5)

    a_val = ValueArray([Value(x) for x in a_np])
    b_val = ValueArray([Value(x) for x in b_np])
    c_val = ValueArray([Value(x) for x in c_np])

    out_val = ((a_val * b_val + c_val) ** 2).relu().sum()
    out_val.backward()

    grad_a_custom = np.array([v.grad for v in a_val.items])
    grad_b_custom = np.array([v.grad for v in b_val.items])
    grad_c_custom = np.array([v.grad for v in c_val.items])

    a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    c_t = torch.tensor(c_np, dtype=torch.float64, requires_grad=True)

    out_t = ((a_t * b_t + c_t) ** 2).relu().sum()
    out_t.backward()

    max_diff_torch = max(
        np.max(np.abs(grad_a_custom - a_t.grad.numpy())),
        np.max(np.abs(grad_b_custom - b_t.grad.numpy())),
        np.max(np.abs(grad_c_custom - c_t.grad.numpy())),
    )

    eps = 1e-6

    def f_eval(a_arr, b_arr, c_arr):
        z = a_arr * b_arr + c_arr
        return float(np.sum(np.maximum(0.0, z ** 2)))

    num_a = np.zeros_like(a_np)
    num_b = np.zeros_like(b_np)
    num_c = np.zeros_like(c_np)
    for i in range(len(a_np)):
        a_p, a_m = a_np.copy(), a_np.copy()
        a_p[i] += eps
        a_m[i] -= eps
        num_a[i] = (f_eval(a_p, b_np, c_np) - f_eval(a_m, b_np, c_np)) / (2 * eps)

        b_p, b_m = b_np.copy(), b_np.copy()
        b_p[i] += eps
        b_m[i] -= eps
        num_b[i] = (f_eval(a_np, b_p, c_np) - f_eval(a_np, b_m, c_np)) / (2 * eps)

        c_p, c_m = c_np.copy(), c_np.copy()
        c_p[i] += eps
        c_m[i] -= eps
        num_c[i] = (f_eval(a_np, b_np, c_p) - f_eval(a_np, b_np, c_m)) / (2 * eps)

    max_diff_num = max(
        np.max(np.abs(grad_a_custom - num_a)),
        np.max(np.abs(grad_b_custom - num_b)),
        np.max(np.abs(grad_c_custom - num_c)),
    )

    assert max_diff_torch < 1e-6, f"PyTorch diff too high: {max_diff_torch}"
    assert max_diff_num < 1e-6, f"Numerical diff too high: {max_diff_num}"

    return {
        "expression": "((a*b + c)**2).relu().sum()",
        "custom_output": out_val.data,
        "torch_output": float(out_t.item()),
        "max_abs_diff_vs_pytorch": float(max_diff_torch),
        "max_abs_diff_vs_numerical": float(max_diff_num),
        "passed_1e_6": bool(max_diff_torch < 1e-6 and max_diff_num < 1e-6),
    }


class CustomMLP:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = [[Value(W1[i, j]) for j in range(8)] for i in range(2)]
        self.b1 = [Value(b1[j]) for j in range(8)]
        self.W2 = [[Value(W2[j, 0])] for j in range(8)]
        self.b2 = Value(b2[0])

    def parameters(self):
        params = []
        for row in self.W1:
            params.extend(row)
        params.extend(self.b1)
        for row in self.W2:
            params.extend(row)
        params.append(self.b2)
        return params

    def forward_one(self, x0, x1):
        h = []
        for j in range(8):
            act = self.b1[j] + x0 * self.W1[0][j] + x1 * self.W1[1][j]
            h.append(act.relu())
        out = self.b2
        for j in range(8):
            out = out + h[j] * self.W2[j][0]
        return out.sigmoid()


def train_custom_mlp(X, y, init_weights, epochs=80, lr=0.1):
    W1, b1, W2, b2 = init_weights
    model = CustomMLP(W1, b1, W2, b2)
    params = model.parameters()
    n = len(X)
    loss_history = []

    t0 = time.perf_counter()
    for _ in range(epochs):
        for p in params:
            p.grad = 0.0

        total_loss = Value(0.0)
        for i in range(n):
            pred = model.forward_one(float(X[i, 0]), float(X[i, 1]))
            yi = float(y[i])
            if yi == 1.0:
                loss_i = -(pred.log())
            else:
                loss_i = -((Value(1.0) - pred).log())
            total_loss = total_loss + loss_i

        mean_loss = total_loss * (1.0 / n)
        mean_loss.backward()

        for p in params:
            p.data -= lr * p.grad

        loss_history.append(mean_loss.data)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    correct = 0
    for i in range(n):
        pred = model.forward_one(float(X[i, 0]), float(X[i, 1])).data
        if (pred >= 0.5) == (y[i] == 1):
            correct += 1
    acc = correct / n
    return elapsed_ms, loss_history[-1], acc, loss_history


def train_torch_mlp(X, y, init_weights, epochs=80, lr=0.1):
    W1, b1, W2, b2 = init_weights
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float32))
        model[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float32))
        model[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_history = []

    t0 = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        preds = model(X_t)
        acc = float(((preds >= 0.5).float() == y_t).float().mean().item())

    return elapsed_ms, loss_history[-1], acc, loss_history


def benchmark_moons_training():
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    rng = np.random.default_rng(42)
    W1 = rng.normal(0.0, 0.5, size=(2, 8))
    b1 = np.zeros(8)
    W2 = rng.normal(0.0, 0.5, size=(8, 1))
    b2 = np.zeros(1)
    init_weights = (W1, b1, W2, b2)

    custom_times, torch_times = [], []
    custom_res = None
    torch_res = None

    for _ in range(5):
        c_ms, c_loss, c_acc, c_hist = train_custom_mlp(X, y, init_weights, epochs=80, lr=0.2)
        t_ms, t_loss, t_acc, t_hist = train_torch_mlp(X, y, init_weights, epochs=80, lr=0.2)
        custom_times.append(c_ms)
        torch_times.append(t_ms)
        custom_res = (c_loss, c_acc, c_hist)
        torch_res = (t_loss, t_acc, t_hist)

    med_custom = statistics.median(custom_times)
    med_torch = statistics.median(torch_times)

    return {
        "n_samples": 100,
        "epochs": 80,
        "runs": 5,
        "custom_median_ms": round(med_custom, 3),
        "pytorch_median_ms": round(med_torch, 3),
        "speedup_pytorch_vs_custom": round(med_custom / med_torch, 2),
        "custom_final_loss": round(custom_res[0], 5),
        "pytorch_final_loss": round(torch_res[0], 5),
        "custom_accuracy": round(custom_res[1], 4),
        "pytorch_accuracy": round(torch_res[1], 4),
        "custom_loss_history": custom_res[2],
        "pytorch_loss_history": torch_res[2],
    }


class SmallBenchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        return self.net(x)


def benchmark_pytorch_internals():
    torch.manual_seed(42)
    model = SmallBenchNet().eval()
    x = torch.randn(512, 64)

    traced_fx = torch.fx.symbolic_trace(model)
    graph_str = str(traced_fx.graph)

    mode_used = "torch.compile"
    try:
        optimized_model = torch.compile(model)
        with torch.no_grad():
            for _ in range(10):
                _ = optimized_model(x)
    except Exception:
        mode_used = "torch.jit.trace"
        optimized_model = torch.jit.trace(model, x)
        with torch.no_grad():
            for _ in range(10):
                _ = optimized_model(x)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    iters = 500
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        eager_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = optimized_model(x)
        opt_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    base_tensor = torch.randn(1024, 1024)
    non_contig_view = base_tensor.t()
    contig_copy = non_contig_view.contiguous()

    tensor_info = {
        "base_shape": list(base_tensor.shape),
        "base_dtype": str(base_tensor.dtype),
        "base_is_contiguous": bool(base_tensor.is_contiguous()),
        "base_stride": list(base_tensor.stride()),
        "view_shape": list(non_contig_view.shape),
        "view_dtype": str(non_contig_view.dtype),
        "view_is_contiguous": bool(non_contig_view.is_contiguous()),
        "view_stride": list(non_contig_view.stride()),
    }

    for _ in range(20):
        _ = (contig_copy * 1.5).reshape(-1).sum()
        _ = (non_contig_view * 1.5).reshape(-1).sum()

    op_iters = 1000
    t0 = time.perf_counter()
    for _ in range(op_iters):
        _ = (contig_copy * 1.5).reshape(-1).sum()
    contig_op_ms = ((time.perf_counter() - t0) / op_iters) * 1000.0

    t0 = time.perf_counter()
    for _ in range(op_iters):
        _ = (non_contig_view * 1.5).reshape(-1).sum()
    non_contig_op_ms = ((time.perf_counter() - t0) / op_iters) * 1000.0

    return {
        "fx_graph": graph_str,
        "execution_mode_tested": mode_used,
        "eager_latency_ms": round(eager_ms, 5),
        "optimized_latency_ms": round(opt_ms, 5),
        "execution_speedup": round(eager_ms / opt_ms, 3),
        "tensor_metadata": tensor_info,
        "contiguous_op_ms": round(contig_op_ms, 5),
        "non_contiguous_op_ms": round(non_contig_op_ms, 5),
        "non_contiguous_slowdown": round(non_contig_op_ms / contig_op_ms, 3),
    }


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    grad_results = verify_gradients()
    print("[Step 4] Gradient Check Passed:", grad_results["passed_1e_6"])
    print(f"  Max diff vs PyTorch:   {grad_results['max_abs_diff_vs_pytorch']:.2e}")
    print(f"  Max diff vs Numerical: {grad_results['max_abs_diff_vs_numerical']:.2e}")

    moons_results = benchmark_moons_training()
    print("\n[Step 5] make_moons Training (Median of 5 runs):")
    print(f"  Custom Autograd: {moons_results['custom_median_ms']} ms | Loss: {moons_results['custom_final_loss']} | Acc: {moons_results['custom_accuracy']}")
    print(f"  PyTorch:         {moons_results['pytorch_median_ms']} ms | Loss: {moons_results['pytorch_final_loss']} | Acc: {moons_results['pytorch_accuracy']}")
    print(f"  PyTorch Speedup: {moons_results['speedup_pytorch_vs_custom']}x")

    plt.figure(figsize=(8, 5))
    plt.plot(moons_results["custom_loss_history"], label="Custom Scalar Autograd", linewidth=2)
    plt.plot(moons_results["pytorch_loss_history"], "--", label="PyTorch Autograd", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("make_moons Training Convergence (2 -> 8 -> 1 MLP)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(results_dir, "loss_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    pt_results = benchmark_pytorch_internals()
    print("\n[Step 6] PyTorch Graph, Execution Mode & Contiguity:")
    print(f"  Eager latency: {pt_results['eager_latency_ms']} ms")
    print(f"  {pt_results['execution_mode_tested']} latency: {pt_results['optimized_latency_ms']} ms (Speedup: {pt_results['execution_speedup']}x)")
    print(f"  Base tensor: shape={pt_results['tensor_metadata']['base_shape']}, dtype={pt_results['tensor_metadata']['base_dtype']}, contiguous={pt_results['tensor_metadata']['base_is_contiguous']}")
    print(f"  Transposed view: shape={pt_results['tensor_metadata']['view_shape']}, dtype={pt_results['tensor_metadata']['view_dtype']}, contiguous={pt_results['tensor_metadata']['view_is_contiguous']}")
    print(f"  Contiguous op: {pt_results['contiguous_op_ms']} ms vs Non-contiguous op: {pt_results['non_contiguous_op_ms']} ms (Slowdown: {pt_results['non_contiguous_slowdown']}x)")

    with open(os.path.join(results_dir, "fx_graph.txt"), "w", encoding="utf-8") as f:
        f.write(pt_results["fx_graph"])

    summary = {
        "gradient_verification": grad_results,
        "moons_benchmark": {k: v for k, v in moons_results.items() if "history" not in k},
        "pytorch_internals": {k: v for k, v in pt_results.items() if k != "fx_graph"},
    }
    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nArtifacts saved to: {results_dir}")


if __name__ == "__main__":
    main()