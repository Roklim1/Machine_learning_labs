# Real-Time F1 Tire Degradation & Pit-Stop Predictor

- **Student Name:** Aliaksandr Rudzenia
- **Student ID:** 72786

## 1. Problem
During a Formula 1 race, tires lose grip over time depending on the compound, track temperature, and driving load. If a driver stays out too long, lap times suddenly drop and they can lose position to an undercut. The goal of this project is to build a small, fast model that uses live car telemetry to estimate tire degradation and predict if a pit stop should happen in the next 1–2 laps.

## 2. Dataset
There will a dataset of clean racing laps from the 2022–2025 seasons using two free sources:
- FastF1 Python API: Lap times, sector times, tire compound (Soft, Medium, Hard), tire age (`TyreLife`), track and air temperatures, and per-lap telemetry averages.
- Kaggle F1 World Championship Dataset: Basic track info to normalize features across different tracks.
Safety car laps, rain sessions, and pit in/out laps will be removed. Data from 2022–2024 will be used for training, and 2025 races will be used for validation and testing.

## 3. Success Metrics
Using a sliding window of the last 5 laps, the model will predict two things:
- Lap-time loss
- Pit-stop window in the next 2 laps

## 4. Deployment Constraints
To run locally on a pit-wall laptop or onboard telemetry unit without cloud delays:
- Latency: <= 50 ms per prediction
- Model size: <= 5 MB on disk

## 5. Planned Optimization
1. Baseline Model: Train a standard MLP / 1D-CNN model on the 5-lap telemetry windows to get strong baseline accuracy.
2. Knowledge Distillation: Train a much smaller student network to mimic the larger model's predictions.
3. INT8 Quantization: Convert the student model to ONNX and apply INT8 post-training quantization to shrink the file size below 1.5 MB and keep CPU inference well under 50 ms.