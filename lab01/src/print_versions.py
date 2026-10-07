import sys
import importlib.metadata

packages = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]

print(f"Python: {sys.version}")

for pkg in packages:
    try:
        ver = importlib.metadata.version(pkg)
        print(f"{pkg}: {ver}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg}: NOT INSTALLED")