"""
Label Generator for GestureTalk Competition Edition
Generates gesture_labels.csv and gesture_labels.npy in root and labels/ directory.
"""

import os
import csv
import numpy as np
from config import GESTURE_CLASSES, LABELS_PATH_CSV, LABELS_PATH_NPY

def generate_labels():
    labels_dir = os.path.dirname(LABELS_PATH_CSV)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # NumPy array format
    labels_arr = np.array(GESTURE_CLASSES)
    np.save(LABELS_PATH_NPY, labels_arr)
    np.save("gesture_labels.npy", labels_arr)
    print(f"Saved NumPy labels to {LABELS_PATH_NPY} and gesture_labels.npy")

    # CSV format: index,label
    for csv_path in [LABELS_PATH_CSV, "gesture_labels.csv"]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label"])
            for idx, label in enumerate(GESTURE_CLASSES):
                writer.writerow([idx, label])
        print(f"Saved CSV labels to {csv_path}")

if __name__ == "__main__":
    generate_labels()
