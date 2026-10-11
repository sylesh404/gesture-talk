"""
Evaluation Module for GestureTalk Competition Edition.
Loads dataset, evaluates gesture_model.keras, and prints Accuracy, Precision, Recall, F1-score, Macro F1, Weighted F1, and Confusion Matrix.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.utils import to_categorical
from config import GESTURE_CLASSES, DATASET_PATH, MODEL_PATH_KERAS, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES

def evaluate_model():
    print("=" * 60)
    print("Evaluating Competition Gesture Model...")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH_KERAS) and not os.path.exists("gesture_model.keras"):
        print("Error: gesture_model.keras not found. Please train the model first.")
        return

    model_path = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else "gesture_model.keras"
    model = tf.keras.models.load_model(model_path)

    X = []
    y_raw = []
    label_map = {label: idx for idx, label in enumerate(GESTURE_CLASSES)}

    for label in GESTURE_CLASSES:
        gesture_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(gesture_dir):
            continue

        seq_dirs = [os.path.join(gesture_dir, d) for d in os.listdir(gesture_dir)
                    if os.path.isdir(os.path.join(gesture_dir, d))]

        for seq_dir in seq_dirs:
            frame_files = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith(".npy")])
            if len(frame_files) != SEQUENCE_LENGTH:
                continue

            frames = []
            valid = True
            for frame_file in frame_files:
                data = np.load(frame_file)
                if data.shape[0] == NUM_LANDMARK_COORDINATES:
                    frames.append(data)
                else:
                    valid = False
                    break

            if valid and len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y_raw.append(label_map[label])

    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)
    y = to_categorical(y_raw, num_classes=len(GESTURE_CLASSES))

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_raw, random_state=42
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro")[2]
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted")[2]
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1-Score:       {macro_f1:.4f}")
    print(f"Weighted F1-Score:    {weighted_f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=GESTURE_CLASSES))

    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }

if __name__ == "__main__":
    evaluate_model()
