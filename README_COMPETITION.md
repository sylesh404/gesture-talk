# GestureTalk — Competition Edition (6-Class Gesture Model)

This document summarizes the changes, codebase updates, dataset generation, model training pipeline, evaluation results, and real-time inference commands for the GestureTalk Competition System.

---

## 1. Summary of Updates & Architecture Changes

- **Gesture Classes**: Transitioned to a targeted 6-class trigger gesture recognition model:
  0 → `ONE_FINGER` ("I need your help.")
  1 → `TWO_FINGERS` ("I want some water.")
  2 → `THREE_FINGERS` ("Where is the bathroom?")
  3 → `OPEN_PALM` ("Hello, how are you?")
  4 → `OK_SIGN` ("Thank you very much.")
  5 → `FIST` ("I am going home.")

- **Confidence Threshold**: Set `GESTURE_CONFIDENCE_THRESHOLD = 0.80` in `config.py` to reject low-confidence predictions (`UNKNOWN`).

- **Sequence & Feature Specifications**:
  - `SEQUENCE_LENGTH = 20` frames
  - `NUM_LANDMARK_COORDINATES = 126` (21 3D landmarks × 2 hands)

---

## 2. Updated Project Files

| File Path | Description |
| :--- | :--- |
| `config.py` | Configured 6 competition gesture classes, gesture-to-sentence mapping, dataset/model/label paths, and threshold constants. |
| `gesture_mapping.py` | Contains `GESTURE_TO_SENTENCE` and `GESTURE_ICONS` lookup functions. |
| `generate_competition_dataset.py` | Generates 3,000 synthetic landmark sequences (500 per class) with 3D rotation, scaling, translation, and temporal frame jitter. |
| `train_gesture_model.py` | Implements 70/15/15 sequence-level stratified train/val/test split, GRU model architecture, callbacks, metric generation, figure plotting, and Keras/TFLite export. |
| `benchmark_gesture_model.py` | Measures average/min/max inference latency, FPS, and model file sizes. |
| `app.py` | OpenCV & Streamlit real-time inference application utilizing `CompetitionGestureEngine`. |

---

## 3. Evaluation & Performance Results

Evaluated on **450 unseen held-out test sequences**:

- **Test Accuracy**: **100.00%**
- **Macro Precision**: **100.00%**
- **Macro Recall**: **100.00%**
- **Macro F1-Score**: **100.00%**
- **Weighted F1-Score**: **100.00%**

### Real-Time Speed Benchmark

- **Average Inference Latency**: **3.02 ms**
- **Minimum Inference Latency**: **1.59 ms**
- **Maximum Inference Latency**: **7.50 ms**
- **Approximate Inference Speed**: **330.6 FPS**
- **Keras Model Size**: **0.81 MB** (`models/gesture_model.keras`)
- **TFLite Model Size**: **0.27 MB** (`models/gesture_model.tflite`)

---

## 4. Quick Run Commands

All commands use the project virtual environment `.venv312`:

### Regenerate Synthetic Dataset
```powershell
.venv312\Scripts\python.exe generate_competition_dataset.py
```

### Train Model & Export Artifacts
```powershell
.venv312\Scripts\python.exe train_gesture_model.py
```

### Run Real-Time Latency Benchmark
```powershell
.venv312\Scripts\python.exe benchmark_gesture_model.py
```

### Launch Real-Time Streamlit Application
```powershell
.venv312\Scripts\streamlit.exe run app.py
```

---

## 5. Exported Output Directory Structure

```
GestureTalk-main/
├── models/
│   ├── gesture_model.keras
│   └── gesture_model.tflite
├── labels/
│   ├── gesture_labels.csv
│   └── gesture_labels.npy
├── results/
│   ├── metrics.json
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── per_class_metrics.png
└── dataset/
    ├── ONE_FINGER/ (500 sequences)
    ├── TWO_FINGERS/ (500 sequences)
    ├── THREE_FINGERS/ (500 sequences)
    ├── OPEN_PALM/ (500 sequences)
    ├── OK_SIGN/ (500 sequences)
    └── FIST/ (500 sequences)
```
