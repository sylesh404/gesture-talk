"""
Evaluation Figures Generator for GestureTalk Competition Edition.
Generates publication-quality charts:
- gesture_confusion_matrix.png
- gesture_per_class_metrics.png
- gesture_overall_metrics.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from evaluate_gesture_model import evaluate_model
from config import GESTURE_CLASSES

def generate_figures():
    results = evaluate_model()
    if not results:
        return

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    cm = results["confusion_matrix"]
    acc = results["accuracy"]

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.title(f"GestureTalk Competition Edition - Confusion Matrix\n(Accuracy: {acc*100:.2f}%)")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("True Gesture")
    plt.tight_layout()
    plt.savefig("gesture_confusion_matrix.png", dpi=300)
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved 'gesture_confusion_matrix.png'")

    # 2. Per-Class Metrics (Precision, Recall, F1-Score)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    x = np.arange(len(GESTURE_CLASSES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, prec, width, label="Precision", color="#4a7c59")
    plt.bar(x, rec, width, label="Recall", color="#e85d2a")
    plt.bar(x + width, f1, width, label="F1-Score", color="#2b580c")

    plt.xlabel("Gesture Classes")
    plt.ylabel("Score")
    plt.title("GestureTalk Competition Edition - Per-Class Metrics")
    plt.xticks(x, GESTURE_CLASSES, rotation=15)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("gesture_per_class_metrics.png", dpi=300)
    plt.savefig("per_class_metrics.png", dpi=300)
    plt.close()
    print("Saved 'gesture_per_class_metrics.png'")

    # 3. Overall Metrics Summary
    macro_f1 = results["macro_f1"]
    weighted_f1 = results["weighted_f1"]

    plt.figure(figsize=(7, 5))
    metrics_names = ["Accuracy", "Macro F1", "Weighted F1"]
    metrics_values = [acc, macro_f1, weighted_f1]
    colors = ["#e85d2a", "#4a7c59", "#1a1208"]

    bars = plt.bar(metrics_names, metrics_values, color=colors, width=0.5)
    plt.ylabel("Score (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    plt.title("GestureTalk Competition Edition - Overall Performance")

    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val*100:.2f}%",
                 ha="center", va="bottom", fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("gesture_overall_metrics.png", dpi=300)
    plt.savefig("overall_metrics.png", dpi=300)
    plt.close()
    print("Saved 'gesture_overall_metrics.png'")

if __name__ == "__main__":
    generate_figures()