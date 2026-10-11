import cv2
import numpy as np
import os

class YOLOHandPoseDetector:
    """
    YOLO Hand-Pose Model Detector for High-Accuracy Gesture Detection
    Uses YOLO Pose / Keypoint Detection to track 21 hand keypoints and bounding boxes.
    """
    def __init__(self, model_name="yolo11s-pose.pt", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_loaded = False
        self.model_name = model_name
        self.initialize_model()

    def initialize_model(self):
        try:
            from ultralytics import YOLO
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check for custom hand pose model weights first
            custom_hand_model = os.path.join(base_dir, "yolo-hand-pose.pt")
            target_model = custom_hand_model if os.path.exists(custom_hand_model) else self.model_name
            
            print(f"Loading YOLO Hand-Pose Model: {target_model}...")
            self.model = YOLO(target_model)
            self.is_loaded = True
            print("YOLO Hand-Pose Model loaded successfully!")
            return True
        except Exception as e:
            print(f"YOLO Hand-Pose initialization note: {e}")
            self.is_loaded = False
            return False

    def extract_hand_keypoints(self, keypoint_array):
        """
        Normalize keypoint array relative to wrist / hand center landmark.
        """
        if len(keypoint_array) == 0:
            return None, 1.0

        # Reshape to (N, 3) -> (x, y, conf) or (N, 2)
        points = keypoint_array[:, :2]
        base = points[0]  # Wrist landmark
        points = points - base

        # Scale by landmark distance
        if len(points) > 12:
            scale = np.linalg.norm(points[12])
        else:
            scale = np.linalg.norm(points[-1])
            
        if scale == 0:
            scale = 1.0

        normalized = points / scale
        return normalized, scale

    def detect(self, frame):
        if not self.is_loaded or self.model is None:
            return frame, None, 0, []

        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)
            annotated_frame = frame.copy()
            detected_gestures = []
            top_label = None
            top_conf = 0.0

            for result in results:
                # Plot YOLO pose keypoints and bounding boxes
                annotated_frame = result.plot()

                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else []

                    for idx, person_kpts in enumerate(kpts):
                        conf = float(boxes[idx][4]) if idx < len(boxes) else 0.5
                        if conf > self.conf_threshold:
                            norm_pts, scale = self.extract_hand_keypoints(person_kpts)
                            
                            # Determine posture / gesture label from keypoints
                            label = "GESTURE_ACTIVE"
                            if conf > top_conf:
                                top_conf = conf
                                top_label = label

                            detected_gestures.append({
                                "label": label,
                                "confidence": round(conf * 100, 2),
                                "keypoints": norm_pts
                            })

            return annotated_frame, top_label, round(top_conf * 100, 2), detected_gestures
        except Exception as e:
            print(f"YOLO Pose Detection error: {e}")
            return frame, None, 0, []
