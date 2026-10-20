import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    num_classes = 10
    y_one_hot = np.eye(num_classes)[y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.30, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

class TwoHiddenLayerNN:
    def __init__(self, input_dim=64, hidden_dim=16, output_dim=10, seed=42):
        np.random.seed(seed)
        self.hidden_dim = hidden_dim
        
        self.W1 = np.random.normal(0, 0.01, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.normal(0, 0.01, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.normal(0, 0.01, (hidden_dim, output_dim))
        self.b3 = np.zeros(output_dim)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.probs = self.softmax(self.Z3)
        return self.probs

    def compute_loss(self, y_true):
        eps = 1e-15
        probs_clipped = np.clip(self.probs, eps, 1 - eps)
        loss = -np.mean(np.sum(y_true * np.log(probs_clipped), axis=1))
        return loss

    def backward(self, y_true):
        N = y_true.shape[0]

        dZ3 = (self.probs - y_true) / N
        self.dW3 = np.dot(self.A2.T, dZ3)
        self.db3 = np.sum(dZ3, axis=0)

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0)
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        self.dW1 = np.dot(self.X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0)

    def update_params(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

def gradient_check(nn, X_sample, y_sample, eps=1e-5):
    nn.forward(X_sample)
    nn.compute_loss(y_sample)
    nn.backward(y_sample)

    weights_dict = {
        'W1': (nn.W1, nn.dW1),
        'W2': (nn.W2, nn.dW2),
        'W3': (nn.W3, nn.dW3)
    }

    rel_errors = []
    print("\n--- Numerical Gradient Checking (10 random weights) ---")
    np.random.seed(42)

    for i in range(10):
        w_name = np.random.choice(['W1', 'W2', 'W3'])
        w_mat, dw_mat = weights_dict[w_name]
        r = np.random.randint(0, w_mat.shape[0])
        c = np.random.randint(0, w_mat.shape[1])

        orig_val = w_mat[r, c]

        w_mat[r, c] = orig_val + eps
        nn.forward(X_sample)
        loss_plus = nn.compute_loss(y_sample)

        w_mat[r, c] = orig_val - eps
        nn.forward(X_sample)
        loss_minus = nn.compute_loss(y_sample)

        w_mat[r, c] = orig_val

        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_analytic = dw_mat[r, c]

        denom = max(abs(grad_num) + abs(grad_analytic), 1e-12)
        rel_error = abs(grad_num - grad_analytic) / denom
        rel_errors.append(rel_error)

        status = "PASS" if rel_error < 1e-6 else "FAIL"
        print(f"Weight {w_name}[{r},{c}]: Num={grad_num:+.6e}, Analytic={grad_analytic:+.6e}, RelError={rel_error:.2e} -> {status}")

    passed = sum(1 for e in rel_errors if e < 1e-6)
    print(f"Gradient Check Result: {passed}/10 passed (Criterion: >= 9/10 with error < 1e-6)")
    return passed >= 9

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_dim, lr, epochs=50, batch_size=32):
    model = TwoHiddenLayerNN(input_dim=64, hidden_dim=hidden_dim, output_dim=10)
    train_losses = []
    N = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            model.forward(X_batch)
            model.compute_loss(y_batch)
            model.backward(y_batch)
            model.update_params(lr)

        model.forward(X_train)
        epoch_loss = model.compute_loss(y_train)
        train_losses.append(epoch_loss)

    test_probs = model.forward(X_test)
    preds = np.argmax(test_probs, axis=1)
    targets = np.argmax(y_test, axis=1)
    acc = np.mean(preds == targets)

    return train_losses, acc

def benchmark_timing(model, X_sample, y_sample, runs=100):
    forward_times = []
    backward_times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        model.forward(X_sample)
        t1 = time.perf_counter()
        model.compute_loss(y_sample)
        model.backward(y_sample)
        t2 = time.perf_counter()

        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)

    median_fwd = np.median(forward_times)
    median_bwd = np.median(backward_times)
    print(f"\nTiming (median of 100 runs on batch size {X_sample.shape[0]}):")
    print(f"Forward pass:  {median_fwd:.4f} ms")
    print(f"Backward pass: {median_bwd:.4f} ms")
    return median_fwd, median_bwd

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Dataset split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    check_model = TwoHiddenLayerNN(hidden_dim=16)
    check_model.forward(X_train)
    untrained_loss = check_model.compute_loss(y_train)
    print(f"Untrained network loss: {untrained_loss:.4f} (Expected ln(10) ≈ 2.3026)")

    out = check_model.forward(X_train[:5])
    print(f"Output shape: {out.shape}, Row sums within 1e-6: {np.allclose(out.sum(axis=1), 1.0, atol=1e-6)}")

    gradient_check(check_model, X_train[:32], y_train[:32])

    widths = [16, 64]
    lrs = [0.001, 0.01, 0.1]

    plt.figure(figsize=(10, 6))
    results = []

    for w in widths:
        for lr in lrs:
            label = f"W={w}, lr={lr}"
            losses, test_acc = train_and_evaluate(X_train, y_train, X_test, y_test, hidden_dim=w, lr=lr)
            plt.plot(losses, label=f"{label} (Acc: {test_acc:.3f})")
            results.append((w, lr, test_acc))
            print(f"Completed {label} -> Test Acc: {test_acc:.4f}")

    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Loss Curves across Learning Rates and Hidden Widths')
    plt.legend()
    plt.grid(True)
    plt.savefig('lab02/results/loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot to results/loss_curves.png")

    bench_model = TwoHiddenLayerNN(hidden_dim=64)
    benchmark_timing(bench_model, X_train[:32], y_train[:32])