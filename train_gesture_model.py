"""
Training Pipeline for GestureTalk Competition Edition.
Trains 6-class competition gesture recognition model with 70/15/15 sequence-level split,
callbacks, augmentation, metric evaluation, figure generation, and Keras/TFLite export.
"""

import os
import json
import csv
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    GESTURE_CLASSES,
    DATASET_PATH,
    MODEL_PATH_KERAS,
    MODEL_PATH_TFLITE,
    LABELS_PATH_CSV,
    LABELS_PATH_NPY,
    SEQUENCE_LENGTH,
    NUM_LANDMARK_COORDINATES
)

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_competition_dataset():
    print("=" * 60)
    print("Loading Competition Dataset...")
    print("=" * 60)

    X = []
    y = []

    label_map = {label: idx for idx, label in enumerate(GESTURE_CLASSES)}

    for label in GESTURE_CLASSES:
        gesture_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(gesture_dir):
            print(f"Warning: Directory '{gesture_dir}' not found.")
            continue

        seq_dirs = [os.path.join(gesture_dir, d) for d in os.listdir(gesture_dir)
                    if os.path.isdir(os.path.join(gesture_dir, d))]

        seq_count = 0
        corrupted_count = 0
        for seq_dir in seq_dirs:
            frame_files = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith(".npy")])

            if len(frame_files) != SEQUENCE_LENGTH:
                corrupted_count += 1
                continue

            frames = []
            valid_seq = True
            for frame_file in frame_files:
                try:
                    data = np.load(frame_file)
                    if data.shape[0] == NUM_LANDMARK_COORDINATES and not np.isnan(data).any():
                        frames.append(data)
                    else:
                        valid_seq = False
                        break
                except Exception:
                    valid_seq = False
                    break

            if valid_seq and len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(label_map[label])
                seq_count += 1
            else:
                corrupted_count += 1

        print(f"Loaded {seq_count} valid sequences for '{label}' (Corrupted/Skipped: {corrupted_count})")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nTotal Dataset Shape: X={X.shape}, y={y.shape}")
    return X, y

def apply_landmark_augmentation(X_data, noise_std=0.005):
    """
    Conservative online landmark augmentation for training sequence data.
    Adds small gaussian noise and slight scaling jitter.
    """
    augmented = np.copy(X_data)
    noise = np.random.normal(0, noise_std, size=augmented.shape).astype(np.float32)
    scale = np.random.uniform(0.95, 1.05, size=(augmented.shape[0], 1, 1)).astype(np.float32)
    return (augmented * scale) + noise

def train_and_export():
    set_seed(42)
    X, y_raw = load_competition_dataset()

    if len(X) == 0:
        raise ValueError("Dataset is empty! Please populate dataset/ first.")

    # 70% Train, 15% Validation, 15% Test Sequence-Level Split with Stratification
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y_raw, test_size=0.30, stratify=y_raw, random_state=42
    )

    X_val, X_test, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Augment training set to double training sample count conservatively
    X_train_aug = apply_landmark_augmentation(X_train_raw)
    X_train = np.concatenate([X_train_raw, X_train_aug], axis=0)
    y_train_raw_full = np.concatenate([y_train_raw, y_train_raw], axis=0)

    y_train = to_categorical(y_train_raw_full, num_classes=len(GESTURE_CLASSES))
    y_val = to_categorical(y_val_raw, num_classes=len(GESTURE_CLASSES))
    y_test = to_categorical(y_test_raw, num_classes=len(GESTURE_CLASSES))

    print("\n" + "=" * 60)
    print("SEQUENCE SPLIT SUMMARY")
    print(f"Train sequences:      {X_train_raw.shape[0]} (augmented to {X_train.shape[0]})")
    print(f"Validation sequences: {X_val.shape[0]}")
    print(f"Test sequences:       {X_test.shape[0]}")
    print("=" * 60)

    # Build GRU Neural Network Model
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES)),
        BatchNormalization(),
        Dropout(0.2),

        GRU(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu"),
        Dense(len(GESTURE_CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(MODEL_PATH_KERAS), exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH_KERAS,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )

    print("\nTraining 6-Class Competition Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Ensure best model is saved locally and at root
    model.save(MODEL_PATH_KERAS)
    model.save("gesture_model.keras")
    print(f"\nSaved Keras Model to '{MODEL_PATH_KERAS}' and 'gesture_model.keras'")

    # Save Labels
    os.makedirs(os.path.dirname(LABELS_PATH_NPY), exist_ok=True)
    np.save(LABELS_PATH_NPY, np.array(GESTURE_CLASSES))
    np.save("gesture_labels.npy", np.array(GESTURE_CLASSES))

    with open(LABELS_PATH_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gesture_label"])
        for idx, label in enumerate(GESTURE_CLASSES):
            writer.writerow([idx, label])

    with open("gesture_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gesture_label"])
        for idx, label in enumerate(GESTURE_CLASSES):
            writer.writerow([idx, label])

    print("Saved gesture labels (CSV & NPY).")

    # Export TFLite Model via SavedModel
    tflite_success = False
    temp_export = "temp_tflite_export"
    if os.path.exists(temp_export):
        shutil.rmtree(temp_export)

    try:
        model.export(temp_export)
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_export)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        with open(MODEL_PATH_TFLITE, "wb") as f:
            f.write(tflite_model)
        with open("gesture_model.tflite", "wb") as f:
            f.write(tflite_model)
        tflite_success = True
        print(f"Exported TFLite Model to '{MODEL_PATH_TFLITE}' and 'gesture_model.tflite'")
    except Exception as e:
        print("Note: TFLite export failed:", e)
    finally:
        if os.path.exists(temp_export):
            shutil.rmtree(temp_export)

    # Output directory for results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Plot Training & Validation Accuracy/Loss Curves
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], "r--", label="Validation Accuracy", linewidth=2)
    plt.title("GestureTalk - Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_accuracy.png"), dpi=300)
    plt.savefig("training_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["loss"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], "r--", label="Validation Loss", linewidth=2)
    plt.title("GestureTalk - Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_loss.png"), dpi=300)
    plt.savefig("training_loss.png", dpi=300)
    plt.close()

    # Evaluation on Unseen Held-out TEST SET ONLY
    eval_loss, eval_acc = model.evaluate(X_test, y_test, verbose=0)
    val_loss = float(np.min(history.history["val_loss"]))
    val_acc = float(np.max(history.history["val_accuracy"]))

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_accuracy = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(precision_recall_fscore_support(y_true, y_pred, average="macro")[2])
    weighted_f1 = float(precision_recall_fscore_support(y_true, y_pred, average="weighted")[2])
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.title(f"GestureTalk Competition Model - Confusion Matrix\n(Test Accuracy: {test_accuracy * 100:.2f}%)")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("True Gesture")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.savefig("gesture_confusion_matrix.png", dpi=300)
    plt.close()

    # Plot Per-Class Metrics
    x_indices = np.arange(len(GESTURE_CLASSES))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - width, prec, width, label="Precision", color="#2b580c")
    plt.bar(x_indices, rec, width, label="Recall", color="#e85d2a")
    plt.bar(x_indices + width, f1, width, label="F1-Score", color="#4a7c59")
    plt.xlabel("Gesture Classes")
    plt.ylabel("Score")
    plt.title("GestureTalk Competition Model - Per-Class Metrics")
    plt.xticks(x_indices, GESTURE_CLASSES, rotation=15)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_class_metrics.png"), dpi=300)
    plt.savefig("per_class_metrics.png", dpi=300)
    plt.savefig("gesture_per_class_metrics.png", dpi=300)
    plt.close()

    # Save metrics.json
    per_class_metrics = {}
    for idx, name in enumerate(GESTURE_CLASSES):
        per_class_metrics[name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1_score": float(f1[idx]),
            "test_samples": int(support[idx])
        }

    metrics_dict = {
        "num_classes": len(GESTURE_CLASSES),
        "total_samples": int(len(X)),
        "train_samples": int(X_train_raw.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": val_acc,
        "best_val_loss": val_loss,
        "test_accuracy": test_accuracy,
        "test_loss": float(eval_loss),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
        "tflite_export": tflite_success
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Print Official Required Report
    print("\n" + "=" * 50)
    print("GESTURETALK MODEL EVALUATION")
    print("=" * 50)
    print(f"Classes: {len(GESTURE_CLASSES)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro Precision: {macro_prec * 100:.2f}%")
    print(f"Macro Recall: {macro_rec * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print("\n" + "=" * 50)
    print("PER-CLASS RESULTS")
    print("=" * 50)
    for idx, name in enumerate(GESTURE_CLASSES):
        print(f"\n{name}")
        print(f"Precision: {prec[idx] * 100:.2f}%")
        print(f"Recall: {rec[idx] * 100:.2f}%")
        print(f"F1: {f1[idx] * 100:.2f}%")
    print("=" * 50 + "\n")

    return metrics_dict

if __name__ == "__main__":
    train_and_export()
