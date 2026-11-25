# Lab 06

## 1. Goal
The goal of this lab was to turn a basic Fashion-MNIST training script into a clean, configurable pipeline with checkpoints and MLflow tracking. We also profiled the training loop to find where time is spent and tested how worker count, batch size, learning-rate schedulers, optimizers, gradient accumulation, and checkpoint resuming affect training speed and accuracy.

## 2. Method
We split the 2-layer CNN into clean modules driven by configs/config.yaml and logged all runs to MLflow. First, we trained the baseline model for 5 epochs and saved checkpoints. Next, we used torch.profiler on 1 epoch and measured speed across num_workers and batch_size. Finally, we compared three LR schedulers and SGD+Momentum vs Adam over 10 epochs, checked if gradient accumulation matches a native 256 batch, and tested stopping training at epoch 3 and resuming from a checkpoint up to epoch 6.

## 3. Results

### Baseline & Profiler Breakdown
- Model Parameters: 20,490
- Baseline Final Val Accuracy (5 epochs): 89.80%
- Baseline Test Loss / Accuracy: 0.2982 / 89.21%

| Stage | Time (s) | Share (%) |
| :--- | :---: | :---: |
| Data Loading (`DataLoader` + transforms) | 13.5228 | 73.61% |
| Backward Pass | 2.5347 | 13.80% |
| Forward Pass | 1.6506 | 8.99% |
| Optimizer Step | 0.6625 | 3.61% |

### DataLoader
| `num_workers` | Epoch 1 Time (s) | Epoch 2 Time (s) | Data Wait (s) | Compute (s) | Throughput (samples/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 21.8501 | 21.6694 | 17.0770 | 4.5923 | 2538.1 |
| 1 | 19.4920 | 19.5696 | 13.5591 | 6.0106 | 2810.5 |
| 2 | 24.3641 | 24.3238 | 17.9929 | 6.3309 | 2261.2 |
| 4 | 41.4392 | 41.6687 | 35.1429 | 6.5258 | 1319.9 |

### Batch-Size Comparison
| Batch Size | Throughput (samples/s) | Peak Memory (MB) | Total Time (s) | Test Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 32 | 1512.1 | 2.182 | 181.864 | 90.20% |
| 64 | 1776.4 | 2.174 | 154.807 | 90.05% |
| 128 | 1877.2 | 2.180 | 146.498 | 89.21% |
| 256 | 2076.0 | 2.198 | 132.467 | 88.87% |
| 512 | 2559.6 | 2.229 | 107.440 | 87.79% |

### LR Schedulers & Optimizers
| Type | Setup | Final Val Loss | Final Val Acc | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Scheduler | `StepLR` (`step=3, gamma=0.5`) | 0.2634 | 90.64% | 90.17% |
| Scheduler | `CosineAnnealingLR` (`T_max=10`) | 0.2614 | 90.72% | 90.35% |
| Scheduler | `ReduceLROnPlateau` (`factor=0.5, patience=1`) | 0.2541 | 90.70% | 90.38% |
| Optimizer | `Adam` (`lr=1e-3` + Cosine) | 0.2654 | 90.48% | 89.93% |
| Optimizer | `SGD+Momentum` (`lr=0.05, m=0.9` + Cosine) | 0.2374 | 91.50% | 91.40% |

### Gradient Accumulation & Checkpoint Resume
| Test | Setup | Final Val Loss | Final Val Acc | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Grad Accum (5 epochs) | Native `batch_size=256` | 0.3095 | 89.26% | 88.87% |
| Grad Accum (5 epochs) | Accumulated `64 x 4` | 0.3099 | 89.22% | 88.87% |
| Resume Check (6 epochs) | Full Run (No Interrupt) | 0.2783 | - | - |
| Resume Check (6 epochs) | Stopped at Epoch 3 -> Resumed | 0.2801 | - | 89.34% |

## 4. Three conclusions
1. Since our 20k-parameter CNN is very fast on CPU, loading and transforming images took 73.61% of the epoch time. Using num_workers=1 gave the best speed, while num_workers=2 or 4 slowed things down on Windows because spawning extra Python processes adds too much overhead.
2. Larger batches sped up training by 69.3% but lowered test accuracy from 90.20% to 87.79% since the model did fewer weight updates in 5 epochs. Switching from Adam to SGD+Momentum with CosineAnnealingLR gave the best test accuracy overall at 91.40%.
3. Accumulating gradients matched the native 256 batch size, and saving full model, optimizer, and scheduler states allowed us to pause at epoch 3 and resume smoothly without any loss jump.