# Lab 05 

## 1. Goal
The goal of this lab was to implement a basic scalar autograd engine from scratch, check its gradient accuracy, and compare its performance with PyTorch. We also inspected how PyTorch handles computation graphs, traced execution, and tensor memory layout.

## 2. Method
We built a 'Value' class in Python that stores a scalar number, its gradient, and references to parent nodes. During the forward pass, addition, multiplication, powers, and ReLU operations build a dynamic computation graph. For the backward pass, the graph is sorted topologically, the output gradient is set to 1.0, and gradients are accumulated ('+=') in reverse order. First, we verified gradients on '((a*b + c)**2).relu().sum()' against PyTorch autograd and numerical finite differences. Second, we trained a small MLP ('2 -> 8 -> 1' with ReLU and sigmoid) on 100 samples of 'make_moons' for 80 epochs using BCE loss, comparing the median runtime of 5 runs between our engine and PyTorch. Finally, we traced a 3-layer network with 'torch.fx', compared 'eager' execution against 'torch.jit.trace' after warm-up, and measured how transposing a 1024x1024 tensor affects operation speed.

## 3. Results

### Gradient Check
- Expression: '((a*b + c)**2).relu().sum()'
- Max diff vs PyTorch: 0.00e+00
- Max diff vs Numerical: 4.32e-10
- Status: Passed (< 1e-6)

### make_moons Training (Median of 5 Runs)
| Engine | Median Time (ms) | Final Loss | Accuracy | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Custom Autograd | 10095.697 | 0.31828 | 0.87 | 1.00x |
| PyTorch | 68.238 | 0.31828 | 0.87 | 147.95x |

### PyTorch Graph & Memory Contiguity
- Base Tensor: shape=[1024, 1024], dtype=torch.float32, contiguous=True
- Transposed View: shape=[1024, 1024], dtype=torch.float32, contiguous=False

| Benchmark | Mode / Tensor Type | Latency (ms) | Comparison |
| :--- | :--- | :---: | :---: |
| Inference (500 iters) | Eager | 1.12462 | Baseline |
| Inference (500 iters) | Traced ('torch.jit.trace') | 1.09488 | 1.027x faster |
| Tensor Op (1000 iters) | Contiguous | 0.82162 | Baseline |
| Tensor Op (1000 iters) | Non-contiguous | 4.14045 | 5.039x slower |

### Artifacts
- results/fx_graph.txt
- results/loss_comparison.png
- results/metrics_summary.json

## 4. Three conclusions
1. Our scalar autograd engine matches PyTorch gradients to 0.00e+00 difference and numerical gradients within 4.32e-10, showing that topological sorting and gradient accumulation work properly.
2. Both models reached the exact same loss and accuracy on 'make_moons', while PyTorch ran faster due to vectorized C++ operations instead of Python loops.
3. Graph tracing slightly sped up inference, while working with a non-contiguous transposed tensor slowed down operations by 5.039x because of strided memory access and extra copying.