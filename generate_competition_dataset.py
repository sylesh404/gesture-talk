"""
Dataset Generator for GestureTalk Competition Edition.
Generates realistic 3D landmark sequence datasets for the 6 competition trigger gestures:
ONE_FINGER, TWO_FINGERS, THREE_FINGERS, OPEN_PALM, OK_SIGN, FIST.
Includes spatial rotations, scale variations, and temporal movement jitter.
"""

import os
import shutil
import numpy as np
from config import GESTURE_CLASSES, DATASET_PATH, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES

def get_base_hand_landmarks(gesture_type: str) -> np.ndarray:
    """
    Constructs base 21 3D landmarks (x, y, z) for right hand for each gesture.
    Landmark indices follow MediaPipe Hand Topology (0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky).
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist
    pts[0] = [0.0, 0.0, 0.0]

    # Base MCP joints (5, 9, 13, 17)
    pts[1] = [0.08, -0.05, 0.0]   # Thumb CMC
    pts[2] = [0.14, -0.10, 0.0]   # Thumb MCP
    pts[3] = [0.18, -0.16, 0.0]   # Thumb IP
    pts[4] = [0.22, -0.22, 0.0]   # Thumb Tip

    pts[5] = [0.06, -0.25, 0.0]   # Index MCP
    pts[9] = [0.01, -0.26, 0.0]   # Middle MCP
    pts[13] = [-0.04, -0.25, 0.0] # Ring MCP
    pts[17] = [-0.08, -0.22, 0.0] # Pinky MCP

    if gesture_type == "ONE_FINGER":
        # Index extended, others folded
        pts[6] = [0.07, -0.38, 0.0]
        pts[7] = [0.08, -0.50, 0.0]
        pts[8] = [0.09, -0.62, 0.0] # Index Tip extended

        # Middle folded
        pts[10] = [0.01, -0.18, 0.05]
        pts[11] = [0.01, -0.12, 0.08]
        pts[12] = [0.01, -0.08, 0.05]

        # Ring folded
        pts[14] = [-0.04, -0.18, 0.05]
        pts[15] = [-0.04, -0.12, 0.08]
        pts[16] = [-0.04, -0.08, 0.05]

        # Pinky folded
        pts[18] = [-0.08, -0.16, 0.05]
        pts[19] = [-0.08, -0.10, 0.08]
        pts[20] = [-0.08, -0.06, 0.05]

    elif gesture_type == "TWO_FINGERS":
        # Index & Middle extended (V-sign)
        pts[6] = [0.09, -0.38, 0.0]
        pts[7] = [0.12, -0.50, 0.0]
        pts[8] = [0.15, -0.62, 0.0] # Index Tip

        pts[10] = [-0.02, -0.38, 0.0]
        pts[11] = [-0.05, -0.50, 0.0]
        pts[12] = [-0.08, -0.62, 0.0] # Middle Tip

        # Ring & Pinky folded
        pts[14] = [-0.04, -0.18, 0.05]
        pts[15] = [-0.04, -0.12, 0.08]
        pts[16] = [-0.04, -0.08, 0.05]

        pts[18] = [-0.08, -0.16, 0.05]
        pts[19] = [-0.08, -0.10, 0.08]
        pts[20] = [-0.08, -0.06, 0.05]

    elif gesture_type == "THREE_FINGERS":
        # Index, Middle, Ring extended
        pts[6] = [0.09, -0.38, 0.0]
        pts[7] = [0.12, -0.50, 0.0]
        pts[8] = [0.14, -0.62, 0.0]

        pts[10] = [0.01, -0.38, 0.0]
        pts[11] = [0.01, -0.51, 0.0]
        pts[12] = [0.01, -0.63, 0.0]

        pts[14] = [-0.06, -0.37, 0.0]
        pts[15] = [-0.08, -0.49, 0.0]
        pts[16] = [-0.10, -0.60, 0.0]

        # Pinky folded
        pts[18] = [-0.08, -0.16, 0.05]
        pts[19] = [-0.08, -0.10, 0.08]
        pts[20] = [-0.08, -0.06, 0.05]

        # Thumb folded
        pts[1] = [0.05, -0.05, 0.02]
        pts[2] = [0.08, -0.08, 0.04]
        pts[3] = [0.07, -0.10, 0.05]
        pts[4] = [0.05, -0.12, 0.04]

    elif gesture_type == "FOUR_FINGERS":
        # Index, Middle, Ring, Pinky extended, Thumb folded
        pts[6] = [0.08, -0.38, 0.0]
        pts[7] = [0.10, -0.50, 0.0]
        pts[8] = [0.12, -0.62, 0.0]

        pts[10] = [0.01, -0.39, 0.0]
        pts[11] = [0.01, -0.52, 0.0]
        pts[12] = [0.01, -0.65, 0.0]

        pts[14] = [-0.06, -0.38, 0.0]
        pts[15] = [-0.08, -0.50, 0.0]
        pts[16] = [-0.10, -0.61, 0.0]

        pts[18] = [-0.12, -0.34, 0.0]
        pts[19] = [-0.15, -0.44, 0.0]
        pts[20] = [-0.18, -0.54, 0.0]

        # Thumb folded tucked in
        pts[1] = [0.05, -0.05, 0.02]
        pts[2] = [0.08, -0.08, 0.04]
        pts[3] = [0.07, -0.10, 0.05]
        pts[4] = [0.05, -0.12, 0.04]

    elif gesture_type == "OPEN_PALM":
        # All 5 fingers extended open
        pts[6] = [0.08, -0.38, 0.0]
        pts[7] = [0.10, -0.50, 0.0]
        pts[8] = [0.12, -0.62, 0.0]

        pts[10] = [0.01, -0.39, 0.0]
        pts[11] = [0.01, -0.52, 0.0]
        pts[12] = [0.01, -0.65, 0.0]

        pts[14] = [-0.06, -0.38, 0.0]
        pts[15] = [-0.08, -0.50, 0.0]
        pts[16] = [-0.10, -0.61, 0.0]

        pts[18] = [-0.12, -0.34, 0.0]
        pts[19] = [-0.15, -0.44, 0.0]
        pts[20] = [-0.18, -0.54, 0.0]

    elif gesture_type == "OK_SIGN":
        # Thumb & Index tips touching (forming circle), Middle/Ring/Pinky extended
        pts[3] = [0.06, -0.22, 0.02]
        pts[4] = [0.04, -0.26, 0.03] # Thumb Tip

        pts[6] = [0.05, -0.28, 0.02]
        pts[7] = [0.04, -0.27, 0.03]
        pts[8] = [0.04, -0.26, 0.03] # Index Tip touching Thumb Tip

        # Middle, Ring, Pinky extended
        pts[10] = [0.01, -0.39, 0.0]
        pts[11] = [0.01, -0.52, 0.0]
        pts[12] = [0.01, -0.64, 0.0]

        pts[14] = [-0.06, -0.38, 0.0]
        pts[15] = [-0.08, -0.49, 0.0]
        pts[16] = [-0.10, -0.60, 0.0]

        pts[18] = [-0.12, -0.34, 0.0]
        pts[19] = [-0.15, -0.44, 0.0]
        pts[20] = [-0.18, -0.53, 0.0]

    elif gesture_type == "FIST":
        # All fingers folded tightly
        pts[6] = [0.06, -0.18, 0.05]
        pts[7] = [0.06, -0.12, 0.08]
        pts[8] = [0.06, -0.08, 0.05]

        pts[10] = [0.01, -0.18, 0.05]
        pts[11] = [0.01, -0.12, 0.08]
        pts[12] = [0.01, -0.08, 0.05]

        pts[14] = [-0.04, -0.18, 0.05]
        pts[15] = [-0.04, -0.12, 0.08]
        pts[16] = [-0.04, -0.08, 0.05]

        pts[18] = [-0.08, -0.16, 0.05]
        pts[19] = [-0.08, -0.10, 0.08]
        pts[20] = [-0.08, -0.06, 0.05]

    return pts

def generate_full_landmark_vector(base_right_hand: np.ndarray) -> np.ndarray:
    """
    Packs left hand (zeros or mirrored) and right hand landmarks into 126D vector.
    """
    left_hand = np.zeros((21, 3), dtype=np.float32) # Left hand empty
    right_hand = base_right_hand.copy()

    # Wrist normalize with invariant scale (Middle MCP joint landmark 9)
    base = right_hand[0]
    scale = np.linalg.norm((right_hand[9] - base)[:2])
    if scale == 0:
        scale = np.linalg.norm(right_hand[9] - base)
    if scale == 0:
        scale = 1.0
    right_hand = (right_hand - base) / scale

    packed = np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1)])
    return packed

def generate_competition_dataset(sequences_per_class: int = 500):
    print("=" * 60)
    print("Generating Competition Dataset for 6 Gestures...")
    print("=" * 60)

    np.random.seed(42)

    # Clean / recreate dataset directory
    if os.path.exists(DATASET_PATH):
        shutil.rmtree(DATASET_PATH)
    os.makedirs(DATASET_PATH, exist_ok=True)

    total_sequences = 0

    for cls in GESTURE_CLASSES:
        cls_dir = os.path.join(DATASET_PATH, cls)
        os.makedirs(cls_dir, exist_ok=True)
        base_landmarks = get_base_hand_landmarks(cls)

        for seq_idx in range(sequences_per_class):
            seq_dir = os.path.join(cls_dir, f"seq_{seq_idx:04d}")
            os.makedirs(seq_dir, exist_ok=True)

            # Sequence-level 3D rotation (pitch, yaw, roll)
            rx, ry, rz = np.random.uniform(-0.35, 0.35, size=3)
            # Rotation matrices
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            rot_matrix = Rz @ Ry @ Rx

            scale_factor = np.random.uniform(0.80, 1.20)
            translation = np.random.uniform(-0.08, 0.08, size=(1, 3))

            # Temporal velocity (moving hand slightly over sequence)
            velocity = np.random.uniform(-0.003, 0.003, size=(1, 3))

            for frame_idx in range(SEQUENCE_LENGTH):
                # Frame-level micro jitter
                jitter = np.random.normal(0, 0.006, size=base_landmarks.shape)
                transformed = (base_landmarks @ rot_matrix.T) * scale_factor + translation + (velocity * frame_idx) + jitter

                vec126 = generate_full_landmark_vector(transformed)
                assert vec126.shape[0] == NUM_LANDMARK_COORDINATES, f"Vector size {vec126.shape[0]} != {NUM_LANDMARK_COORDINATES}"
                np.save(os.path.join(seq_dir, f"frame_{frame_idx:02d}.npy"), vec126)

            total_sequences += 1

        print(f"Generated {sequences_per_class} sequences for class: {cls}")

    print("=" * 60)
    print(f"Successfully generated {total_sequences} total sequences in '{DATASET_PATH}'!")
    print("=" * 60)

if __name__ == "__main__":
    generate_competition_dataset(sequences_per_class=200)

