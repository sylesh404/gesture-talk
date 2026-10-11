# GestureTalk — Competition Edition: Real-Time Gesture-to-Sentence Recognition, Translation & Speech System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11%20Pose-yellow.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-teal.svg)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100.00%25-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-3.30ms-brightgreen.svg)]()
[![Translation](https://img.shields.io/badge/Translation-20%20Languages-blueviolet.svg)]()

---

**GestureTalk — Competition Edition** is a high-reliability, ultra-low-latency real-time web application engineered for competition demonstrations. It recognizes six distinct trigger hand gestures from a webcam stream, deterministically maps each gesture directly to a complete predefined English sentence, translates the sentence into **20 regional and international languages**, and vocalizes the output using AI Text-to-Speech (TTS).

> [!NOTE]
> **Design Disclaimer**: The six hand gestures (`ONE_FINGER`, `TWO_FINGERS`, `THREE_FINGERS`, `OPEN_PALM`, `OK_SIGN`, `FIST`) serve as a competition trigger-control interface designed for maximum demonstration reliability, low latency, and zero word-assembly friction. They are custom predefined trigger gestures for the prototype demonstration.

---

## 📸 Interface & Workflow Overview

| Competition Web Interpreter UI | Product Overview Landing Page |
| :---: | :---: |
| Real-time gesture emoji tracking, confidence meter, hold progress bar, canonical sentence box, target language selector, and audio speech. | Competition edition product overview detailing system architecture, trigger gesture mapping, performance metrics, and live demo access. |

---

## 🌟 Key Features & Technical Highlights

### 🖐️ Dual-Model Hand Tracking & Keypoint Extraction
- **YOLO11 Pose Detector**: Tracks hand bounding boxes and keypoint locations under varying camera distances and lighting conditions (`yolo11s-pose.pt`).
- **MediaPipe 3D Landmark Solution**: Extracts 21 3D coordinates per hand ($21 \times 3 = 63$ coordinates per hand, 126 coordinates total).

### 📐 Dynamic Wrist Landmark Normalization
- Centers coordinates relative to the wrist base landmark ($LM_0$) and scales coordinates by the middle finger MCP landmark ($LM_{12}$).
- Supports Left-Hand, Right-Hand, and Dual-Hand gestures effortlessly.

### ⏱️ Temporal Sequence GRU Classifier & Rejection System
- Processes 20-frame spatial-temporal sliding windows ($20 \times 126$).
- Built with a 2-layer stacked **GRU (Gated Recurrent Unit)** network featuring **Batch Normalization** and **Dropout**.
- **Confidence Rejection System**: Evaluates predictions against `GESTURE_CONFIDENCE_THRESHOLD = 0.80`. Predictions below 80% are rejected as `UNKNOWN`.

### 🛡️ Debouncing & Cooldown Mechanism
- Requires consecutive frame holding (`HOLD_FRAMES = 8`) to confirm a gesture.
- Implements a post-confirmation debounce cooldown (`COOLDOWN_SECONDS = 1.5`) and automatic hand-release / transition tracking, preventing repeated speech loops.

### ⚡ Sub-4ms Tensor Graph Inference Acceleration
- Uses TensorFlow compiled graph execution `@tf.function(reduce_retracing=True)`.
- Achieves **3.30 ms** inference latency per prediction (a **37.8x speedup** over standard `model.predict()`).

### 🔊 Multilingual Translation & Dual Speech Synthesis
- Translates canonical English sentences into **20 Indian regional and international languages** via `deep-translator`.
- Vocalizes translated sentences using server-side `gTTS` MP3 audio streaming with browser Web Speech API fallback.

---

## 📚 Competition Trigger Gestures & Sentence Mapping

| Gesture Name | Emoji | Predefined Canonical English Sentence | Baseline / State |
| :--- | :---: | :--- | :--- |
| `ONE_FINGER` | ☝️ | `"I need your help."` | Active Trigger |
| `TWO_FINGERS` | ✌️ | `"I want some water."` | Active Trigger |
| `THREE_FINGERS` | 🤟 | `"Where is the bathroom?"` | Active Trigger |
| `OPEN_PALM` | ✋ | `"Hello, how are you?"` | Active Trigger |
| `OK_SIGN` | 👌 | `"Thank you very much."` | Active Trigger |
| `FIST` | ✊ | `"I am going home."` | Active Trigger |
| `UNKNOWN` | ❓ | *Rejection / Idle baseline* | Idle / Rejection |

---

## 🏗️ System Architecture & Data Flow

```
                          ┌───────────────────────────┐
                          │   Live Webcam Stream      │
                          │   (640x480 @ 30 FPS)      │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  OpenCV Frame Capture &   │
                          │  Horizontal Mirroring     │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│  YOLO11 Pose Detector     │                         │  MediaPipe Hand Solution  │
│  (Hand Bounding Boxes)    │                         │  (21 3D Landmarks / Hand) │
└────────────┬──────────────┘                         └────────────┬──────────────┘
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Dynamic Wrist Normalization│
                          │ (Center Wrist & Scale LM) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ 20-Frame Temporal Queue   │
                          │ (Sliding Window: 20x126)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  Fast Tensor GRU Model    │
                          │  Compiled Inference Graph │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Confidence Threshold >80% │
                          │ & Hold Counter (8 frames) │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│  Confidence < 80%         │                         │  Confidence ≥ 80% & Hold  │
│  Status: UNKNOWN / IDLE   │                         │  Status: CONFIRMED        │
└───────────────────────────┘                         └────────────┬──────────────┘
                                                                   │
                                                                   ▼
                                                      ┌───────────────────────────┐
                                                      │ Canonical Sentence Lookup │
                                                      │ (config.GESTURE_TO_SENT) │
                                                      └────────────┬──────────────┘
                                                                   │
                                                                   ▼
                                                      ┌───────────────────────────┐
                                                      │ 20-Language Translator    │
                                                      │ & gTTS Speech Synthesizer │
                                                      └───────────────────────────┘
```

---

## 🛠️ Tech Stack Breakdown

| Area | Technology | Implementation Details |
| :--- | :--- | :--- |
| **Core Backend** | Python 3.10 / 3.12 | Flask web framework, multi-threaded engine. |
| **Computer Vision** | OpenCV 4.9 & MediaPipe | Frame capture, 3D hand landmark extraction. |
| **Object Detection** | Ultralytics YOLO11 Pose | High-precision hand keypoint localization (`yolo11s-pose.pt`). |
| **Deep Learning** | TensorFlow 2.16 / Keras | 2-layer stacked GRU model (`gesture_model.keras`). |
| **Inference Acceleration**| `@tf.function` Compiled Graph | Reduced prediction latency to 3.30 ms per window. |
| **Translation** | `deep-translator` | Neural translation into 20 languages. |
| **Audio & Speech** | `gTTS` & Web Speech API | Automated MP3 synthesis & browser speech fallback. |
| **Frontend UI** | HTML5, CSS3, JavaScript | Modern responsive Competition UI + Landing page. |
| **Testing & Evaluation**| scikit-learn, Matplotlib, Seaborn | Unit tests, metrics evaluation, figure generation. |

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
- **Python 3.10 or 3.12** installed
- **Webcam** connected
- Active internet connection (for online translation and gTTS voice output)

### 2. Clone the Repository
```bash
git clone https://github.com/sharonmary15/GestureTalk.git
cd GestureTalk
```

### 3. Create & Activate Virtual Environment
```cmd
python -m venv .venv312
.\.venv312\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```

### 6. Access in Browser
Navigate to **`http://127.0.0.1:5000`** in your browser:
- Click **Launch Competition Interpreter**
- Click **▶ Start Camera** and perform any of the 6 trigger gestures.

---

## 📂 Project Directory Structure

```
GestureTalk/
│
├── app.py                      # Main Flask application & CompetitionGestureEngine
├── config.py                   # Centralized thresholds, mappings, and class configurations
├── gesture_mapping.py          # Helper functions for sentence and icon lookups
├── setup_labels.py             # CSV and NumPy label generation script
├── yolo_hand_pose.py           # YOLO11 Pose detector integration module
├── yolo_detector.py            # OpenCV Darknet YOLOv3-Tiny fallback module
├── download_yolo_weights.py    # Weights download helper script
│
├── models/
│   └── gesture_model.keras     # Active 6-class competition GRU model
│
├── labels/
│   ├── gesture_labels.csv      # CSV mapping for 6 competition classes
│   └── gesture_labels.npy      # NumPy binary array of class names
│
├── legacy_model/               # Archived legacy 28-class model & label backup
│   ├── isl_lstm_model.keras
│   ├── labels.npy
│   └── labels.csv
│
├── dataset/                    # 6-class temporal sequence dataset (360 sequences)
│   ├── ONE_FINGER/
│   ├── TWO_FINGERS/
│   ├── THREE_FINGERS/
│   ├── OPEN_PALM/
│   ├── OK_SIGN/
│   └── FIST/
│
├── collect_gesture_sequences.py# Interactive sequence collection tool with countdown
├── collect_sequences.py        # Wrapper script for data collection
├── train_gesture_model.py      # GRU neural network training pipeline
├── train_lstm_model.py         # Wrapper script for model training
├── evaluate_gesture_model.py   # Metrics & confusion matrix evaluation tool
├── check_accuracy.py           # Wrapper script for model accuracy checking
├── benchmark_speed.py          # Inference speed benchmarking tool
├── generate_results_figures.py # Plots gesture_confusion_matrix.png & metrics charts
│
├── tests/
│   └── test_competition.py     # Automated unit test suite
│
├── templates/
│   ├── index.html              # Competition Interpreter Web Application
│   └── landing.html            # Competition Product Landing Page
│
├── gesture_confusion_matrix.png# Generated confusion matrix figure
├── gesture_per_class_metrics.png# Generated per-class Precision/Recall/F1 chart
├── gesture_overall_metrics.png # Generated overall performance summary chart
└── requirements.txt            # Python dependencies manifest
```

---

## 📊 Benchmark & Evaluation Results

### Model Performance Metrics
- **Overall Test Accuracy**: **100.00%**
- **Macro F1-Score**: **1.0000**
- **Weighted F1-Score**: **1.0000**

```
               precision    recall  f1-score   support

   ONE_FINGER       1.00      1.00      1.00        12
  TWO_FINGERS       1.00      1.00      1.00        12
THREE_FINGERS       1.00      1.00      1.00        12
    OPEN_PALM       1.00      1.00      1.00        12
      OK_SIGN       1.00      1.00      1.00        12
         FIST       1.00      1.00      1.00        12

     accuracy                           1.00        72
    macro avg       1.00      1.00      1.00        72
 weighted avg       1.00      1.00      1.00        72
```

### Latency Speedup
- **Standard Keras `predict()` Latency**: 124.75 ms / prediction
- **Fast Compiled Tensor Graph Latency**: **3.30 ms** / prediction
- **Inference Speedup Factor**: **37.8x faster**

---

## 🌐 Supported Translation Languages (20 Languages)

English · Hindi · Marathi · Bengali · Tamil · Malayalam · Gujarati · Urdu · Punjabi · Nepali · Spanish · French · German · Arabic · Chinese · Japanese · Korean · Thai · Russian · Portuguese

---

## 📜 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Renders product landing page (`landing.html`). |
| `/app` | `GET` | Renders competition interpreter application (`index.html`). |
| `/video_feed` | `GET` | Multipart MJPEG continuous webcam video stream. |
| `/api/start_camera` | `POST` | Initializes camera capture and starts processing loop. |
| `/api/stop_camera` | `POST` | Releases camera device and stops engine loop. |
| `/api/state` | `GET` | Polls current gesture, confidence, hold progress, sentence, translation, and status. |
| `/api/set_language` | `POST` | Updates target translation language (`{"language": "Hindi"}`). |
| `/api/reset_state` | `POST` | Clears confirmation state and resets debounce tracker. |
| `/api/translate` | `POST` | Translates sentence into target language (`{"sentence": "...", "language": "Hindi"}`). |
| `/api/speak` | `POST` | Generates and streams gTTS MP3 audio file. |

---

## 🧪 Automated Unit Tests

Run the test suite to verify configuration mappings, confidence thresholds, and gesture lookups:
```bash
python tests/test_competition.py
```

---

## 👤 Author & Acknowledgments

* **Sharonmary Lazer Baptist Anthony**
  * B.Sc. Data Science, Mulund College of Commerce (Autonomous), University of Mumbai
  * **Project Guide**: Dr. Priti Pathak

---
*Built for high-reliability competition demonstrations and inclusive AI accessibility.*


