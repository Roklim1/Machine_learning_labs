## 1. Goal
The goal of this lab was to establish a Python environment with strict dependencies, train models on the given dataset, measure key system-level metrics, and evaluate their deployment suitability against Cloud, Edge, Mobile, and TinyML resource budgets.
## 2. Method
In this lab, Python 3.11.8 was used in a virtual environment with all libraries from requirements.txt. Random seed of 42 was created using random, numpy, and torch to ensure reproducible results. The Breast Cancer Wisconsin dataset was split 70/30 with stratification. After training two models: Logistic Regression and Random Forest, it was possible to measure accuracy, training time, single-sample inference latency, model size, and peak RAM usage. 
## 3. Results table or plot
### Baseline Accuracy
| Model | Test Accuracy |
| :--- | :--- |
| LogisticRegression | 0.9474 |
| RandomForest | 0.9357 |

### System Metrics
| Model | Accuracy | Train Time (s) | Inference Latency (ms) | Size (Bytes) | Size (KB) | Peak Memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | 0.9474 | 0.69250 | 0.0708 | 1055 | 1.03 | 267.73 |
| RandomForest | 0.9357 | 0.21219 | 3.5399 | 290889 | 284.07 | 268.98 |

### Resource Budgets
| Platform | Constraints | Logistic Regression | Random Forest | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Cloud | RAM >= 1 GB, Latency <= 100 ms, Size <= 500 MB | Yes | Yes | Fits easily. |
| Edge | RAM 256-1024 MB, Latency <= 50 ms, Size <= 50 MB | Yes | Yes | Fast and light enough. |
| Mobile | RAM 64-256 MB, Latency <= 20 ms, Size <= 10 MB | Yes | Yes | Both fit mobile limits. |
| TinyML | RAM <= 256 KB, Latency <= 10 ms, Size <= 100 KB | Yes | No | Random Forest (284 KB) is too large for the 100 KB limit. |
## 4. Three conclusions
1) Logistic Regression performed slightly better than Random Forest on this dataset and was about 50 times faster in inference.
2) Random Forest requires much more disk storage because it saves 100 decision trees.
3) Both models fit Cloud, Edge, and Mobile platforms. But only Logistic Regression meets the TinyML limits because Random Forest exceeds the 100 KB size ceiling.