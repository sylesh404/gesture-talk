import os
import numpy as np

dataset_path = "dataset"

total_sequences = 0
frames_per_sequence = None
features_per_frame = None

print("Sequences per gesture:\n")

for gesture in sorted(os.listdir(dataset_path)):
    gesture_path = os.path.join(dataset_path, gesture)

    if not os.path.isdir(gesture_path):
        continue

    sequences = [s for s in os.listdir(gesture_path) if os.path.isdir(os.path.join(gesture_path, s))]

    print(f"{gesture}: {len(sequences)} sequences")

    total_sequences += len(sequences)

    if len(sequences) > 0 and frames_per_sequence is None:
        seq_path = os.path.join(gesture_path, sequences[0])

        frames = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
        frames_per_sequence = len(frames)

        sample_file = os.path.join(seq_path, frames[0])
        data = np.load(sample_file)
        features_per_frame = data.shape[0]

print("\n----------------------")
print("Total sequences:", total_sequences)
print("Frames per sequence:", frames_per_sequence)
print("Features per frame:", features_per_frame)