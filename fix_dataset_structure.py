import os
import numpy as np
import shutil

DATASET_PATH = "dataset"
SEQUENCE_LENGTH = 20

for gesture in os.listdir(DATASET_PATH):

    gesture_path = os.path.join(DATASET_PATH, gesture)

    if not os.path.isdir(gesture_path):
        continue

    files = sorted(os.listdir(gesture_path))

    # Check if dataset already has sequence folders
    if len(files) > 0 and os.path.isdir(os.path.join(gesture_path, files[0])):
        print(f"{gesture} already structured. Skipping.")
        continue

    print(f"Fixing {gesture}...")

    npy_files = [f for f in files if f.endswith(".npy")]

    sequence_count = 0

    for i in range(0, len(npy_files), SEQUENCE_LENGTH):

        sequence = npy_files[i:i+SEQUENCE_LENGTH]

        if len(sequence) < SEQUENCE_LENGTH:
            continue

        sequence_folder = os.path.join(gesture_path, str(sequence_count))
        os.makedirs(sequence_folder, exist_ok=True)

        for frame_idx, file in enumerate(sequence):

            src = os.path.join(gesture_path, file)
            dst = os.path.join(sequence_folder, f"{frame_idx}.npy")

            shutil.move(src, dst)

        sequence_count += 1

    print(f"{gesture}: created {sequence_count} sequences")

print("\nDataset conversion complete.")