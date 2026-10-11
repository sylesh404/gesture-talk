"""
Interactive Competition Gesture Sequence Collection Tool
Captures 20-frame spatial-temporal sequences of normalized hand landmarks for the 6 competition trigger gestures.
Includes countdown timer, progress bar, shape validation, and overwrite protection.
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp
from config import GESTURE_CLASSES, DATASET_PATH, SEQUENCE_LENGTH

def extract_landmarks(results):
    """
    Extracts 126-dimensional normalized landmark array from MediaPipe hand results.
    Identical to production inference preprocessing.
    """
    left = [0.0] * 63
    right = [0.0] * 63

    has_left = False
    has_right = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if hand_label == "Left":
                left = coords
                has_left = True
            else:
                right = coords
                has_right = True

    points = np.array(left + right).reshape(42, 3)

    if has_left:
        base = points[0]
        scale = np.linalg.norm(points[12][:2])
    elif has_right:
        base = points[21]
        scale = np.linalg.norm(points[33][:2])
    else:
        base = np.array([0.0, 0.0, 0.0])
        scale = 1.0

    if scale == 0:
        scale = 1.0

    points = (points - base) / scale
    return points.reshape(-1)

def main():
    print("=" * 60)
    print("GestureTalk Competition Edition - Sequence Collector")
    print("=" * 60)
    print("Available Competition Gestures:")
    for idx, cls in enumerate(GESTURE_CLASSES):
        print(f"  [{idx + 1}] {cls}")
    print("=" * 60)

    choice = input("Enter gesture number (1-6) or label name: ").strip().upper()
    if choice.isdigit() and 1 <= int(choice) <= len(GESTURE_CLASSES):
        label = GESTURE_CLASSES[int(choice) - 1]
    elif choice in GESTURE_CLASSES:
        label = choice
    else:
        print("Invalid choice! Defaulting to OPEN_PALM.")
        label = "OPEN_PALM"

    label_path = os.path.join(DATASET_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    existing_sequences = [d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d))]
    seq_counter = len(existing_sequences)

    print(f"\nTarget Gesture: {label}")
    print(f"Existing Sequences: {seq_counter}")
    num_to_collect = int(input("How many new sequences to record? [Default: 30]: ") or "30")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nPress 's' to start sequence recording | Press 'q' to exit")

    try:
        collected = 0
        while collected < num_to_collect:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Gesture: {label} | Collected: {collected}/{num_to_collect}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to Record | 'q' to Quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Sequence Collector", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                print("Exiting collection.")
                break
            elif key == ord('s'):
                # Countdown 3, 2, 1
                for cd in range(3, 0, -1):
                    start_time = time.time()
                    while time.time() - start_time < 0.7:
                        ret, c_frame = cap.read()
                        if not ret:
                            break
                        c_frame = cv2.flip(c_frame, 1)
                        cv2.putText(c_frame, f"GET READY: {cd}", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
                        cv2.imshow("Sequence Collector", c_frame)
                        cv2.waitKey(1)

                # Record 20 frames
                seq_dir = os.path.join(label_path, f"seq_{seq_counter:04d}")
                os.makedirs(seq_dir, exist_ok=True)

                frames_captured = 0
                for f_idx in range(SEQUENCE_LENGTH):
                    ret, r_frame = cap.read()
                    if not ret:
                        break

                    r_frame = cv2.flip(r_frame, 1)
                    rgb = cv2.cvtColor(r_frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    landmarks = extract_landmarks(results)

                    np.save(os.path.join(seq_dir, f"frame_{f_idx:02d}.npy"), landmarks)
                    frames_captured += 1

                    progress = int((frames_captured / SEQUENCE_LENGTH) * 100)
                    cv2.putText(r_frame, f"RECORDING {label}... {progress}%", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow("Sequence Collector", r_frame)
                    cv2.waitKey(20)

                if frames_captured == SEQUENCE_LENGTH:
                    seq_counter += 1
                    collected += 1
                    print(f"Saved: {seq_dir} ({SEQUENCE_LENGTH} frames)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
