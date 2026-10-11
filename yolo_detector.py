import cv2
import numpy as np
import os
from download_yolo_weights import setup_yolo_files, CFG_FILE, WEIGHTS_FILE, NAMES_FILE

class YOLOv3GestureDetector:
    def __init__(self, conf_threshold=0.4, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = []
        self.output_layers = []
        self.is_loaded = False
        
        self.initialize_network()

    def initialize_network(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for custom gesture weights first (e.g. gesture_yolov3.weights / obj.weights)
        custom_weights = os.path.join(base_dir, "gesture_yolov3.weights")
        custom_cfg = os.path.join(base_dir, "gesture_yolov3.cfg")
        custom_names = os.path.join(base_dir, "gesture.names")

        if os.path.exists(custom_weights) and os.path.exists(custom_cfg):
            weights_path = custom_weights
            cfg_path = custom_cfg
            names_path = custom_names if os.path.exists(custom_names) else NAMES_FILE
            print("Loading custom gesture YOLOv3 model...")
        else:
            # Ensure standard YOLOv3-tiny files are downloaded
            setup_yolo_files()
            weights_path = WEIGHTS_FILE
            cfg_path = CFG_FILE
            names_path = NAMES_FILE
            print("Loading YOLOv3-tiny model via OpenCV DNN...")

        if not (os.path.exists(weights_path) and os.path.exists(cfg_path)):
            print(f"YOLOv3 files missing: {cfg_path}, {weights_path}")
            return False

        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names = self.net.getLayerNames()
            try:
                out_layers = self.net.getUnconnectedOutLayers()
                if out_layers.ndim == 1:
                    self.output_layers = [layer_names[i - 1] for i in out_layers]
                else:
                    self.output_layers = [layer_names[i[0] - 1] for i in out_layers]
            except Exception:
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            if os.path.exists(names_path):
                with open(names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                self.classes = ["Gesture"]

            self.is_loaded = True
            print(f"YOLOv3 OpenCV network loaded successfully! Classes: {len(self.classes)}")
            return True
        except Exception as e:
            print(f"Error loading YOLOv3 model in OpenCV: {e}")
            self.is_loaded = False
            return False

    def detect(self, frame):
        if not self.is_loaded or self.net is None:
            return frame, None, 0, []

        h, w, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)

                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)

                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detected_objects = []
        top_label = None
        top_conf = 0

        if len(indices) > 0:
            flattened_indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]

            for i in flattened_indices:
                x, y, dw, dh = boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                label_name = self.classes[class_id] if class_id < len(self.classes) else f"Class {class_id}"

                detected_objects.append({
                    "label": label_name,
                    "confidence": round(conf * 100, 2),
                    "box": [x, y, dw, dh]
                })

                if conf > top_conf:
                    top_conf = conf
                    top_label = label_name

                # Draw bounding box and label overlay on OpenCV frame
                color = (42, 93, 232)  # BGR
                cv2.rectangle(frame, (x, y), (x + dw, y + dh), color, 2)
                
                text = f"{label_name.upper()} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
                cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, top_label, round(top_conf * 100, 2), detected_objects
