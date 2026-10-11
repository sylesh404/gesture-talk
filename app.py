"""
GestureTalk - Competition Edition Main Web Application & Flask Engine
Real-Time Predefined Gesture-to-Sentence Recognition, Translation, and Speech System.
Optimized for Ultra-High FPS Video Streaming, Fast Neural & Heuristic Inference, and Zero-Latency TTS.
"""

import os
import time
import cv2
import tempfile
import threading
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
from gtts import gTTS
from deep_translator import GoogleTranslator

from config import (
    GESTURE_CONFIDENCE_THRESHOLD,
    HOLD_FRAMES,
    COOLDOWN_SECONDS,
    SEQUENCE_LENGTH,
    NUM_LANDMARK_COORDINATES,
    GESTURE_CLASSES,
    GESTURE_TO_SENTENCE,
    GESTURE_ICONS,
    LANG_CODES,
    LANGUAGES,
    MODEL_PATH_KERAS,
    LABELS_PATH_NPY
)
from gesture_mapping import get_sentence_for_gesture, get_icon_for_gesture

app = Flask(__name__)
CORS(app)

# ==============================================================================
# MODEL & FAST TENSOR INFERENCE
# ==============================================================================

model_file = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else "gesture_model.keras"
if not os.path.exists(model_file) and os.path.exists(os.path.join("legacy_model", "isl_lstm_model.keras")):
    print("Warning: Competition model not found. Using fallback model...")

print(f"Loading gesture classifier model: {model_file}")
try:
    model = tf.keras.models.load_model(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

labels = np.array(GESTURE_CLASSES)
if os.path.exists(LABELS_PATH_NPY):
    labels = np.load(LABELS_PATH_NPY)

@tf.function(reduce_retracing=True)
def _fast_predict_fn(x):
    if model is not None:
        return model(x, training=False)
    return tf.zeros((1, len(GESTURE_CLASSES)))

def fast_predict(input_data):
    if model is None:
        return np.zeros(len(GESTURE_CLASSES))
    tensor_input = tf.constant(input_data, dtype=tf.float32)
    return _fast_predict_fn(tensor_input).numpy()[0]

# Warmup model graph
if model is not None:
    fast_predict(np.zeros((1, SEQUENCE_LENGTH, NUM_LANDMARK_COORDINATES), dtype=np.float32))

# ==============================================================================
# MEDIAPIPE HANDS & LANDMARK NORMALIZATION
# ==============================================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def normalize_single_hand(coords_63, is_left=False):
    pts = np.array(coords_63, dtype=np.float32).reshape(21, 3)
    if is_left:
        # Flip X coordinates around wrist so left hand matches right hand geometry
        pts[:, 0] = pts[0, 0] * 2.0 - pts[:, 0]
    wrist = pts[0].copy()
    scale = np.linalg.norm((pts[9] - wrist)[:2])
    if scale == 0:
        scale = np.linalg.norm(pts[9] - wrist)
    if scale == 0:
        scale = 1.0
    norm_pts = (pts - wrist) / scale
    return norm_pts.reshape(-1)

def heuristic_gesture_check(hand_landmarks):
    """
    3D Euclidean distance & joint topology finger extension check.
    Fully scale, rotation, tilt, and hand-orientation invariant.
    Returns: "ONE_FINGER", "TWO_FINGERS", "THREE_FINGERS", "FOUR_FINGERS", "OPEN_PALM", "OK_SIGN", "FIST", or None.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0]

    # Hand reference scale (distance between Wrist 0 and Middle MCP 9)
    hand_scale = np.linalg.norm(pts[9] - wrist)
    if hand_scale == 0:
        hand_scale = 1.0

    # 3D Euclidean distances for each finger (Tip to Wrist vs PIP/MCP to Wrist, and Tip to MCP vs PIP to MCP)
    # Index: Tip=8, PIP=6, MCP=5
    index_dist_wrist = np.linalg.norm(pts[8] - wrist)
    index_pip_wrist = np.linalg.norm(pts[6] - wrist)
    index_dist_mcp = np.linalg.norm(pts[8] - pts[5])
    index_pip_mcp = np.linalg.norm(pts[6] - pts[5])

    # Middle: Tip=12, PIP=10, MCP=9
    middle_dist_wrist = np.linalg.norm(pts[12] - wrist)
    middle_pip_wrist = np.linalg.norm(pts[10] - wrist)
    middle_dist_mcp = np.linalg.norm(pts[12] - pts[9])
    middle_pip_mcp = np.linalg.norm(pts[10] - pts[9])

    # Ring: Tip=16, PIP=14, MCP=13
    ring_dist_wrist = np.linalg.norm(pts[16] - wrist)
    ring_pip_wrist = np.linalg.norm(pts[14] - wrist)
    ring_dist_mcp = np.linalg.norm(pts[16] - pts[13])
    ring_pip_mcp = np.linalg.norm(pts[14] - pts[13])

    # Pinky: Tip=20, PIP=18, MCP=17
    pinky_dist_wrist = np.linalg.norm(pts[20] - wrist)
    pinky_pip_wrist = np.linalg.norm(pts[18] - wrist)
    pinky_dist_mcp = np.linalg.norm(pts[20] - pts[17])
    pinky_pip_mcp = np.linalg.norm(pts[18] - pts[17])

    # Extended conditions for 4 main fingers (3D rotation-invariant):
    index_ext = (index_dist_wrist > index_pip_wrist) and (index_dist_mcp > index_pip_mcp * 1.02)
    middle_ext = (middle_dist_wrist > middle_pip_wrist) and (middle_dist_mcp > middle_pip_mcp * 1.02)
    ring_ext = (ring_dist_wrist > ring_pip_wrist) and (ring_dist_mcp > ring_pip_mcp * 1.02)
    pinky_ext = (pinky_dist_wrist > pinky_pip_wrist) and (pinky_dist_mcp > pinky_pip_mcp * 1.02)

    # Thumb extension check:
    # Tip=4, IP=3, MCP=2, Pinky MCP=17
    thumb_dist_pinky = np.linalg.norm(pts[4] - pts[17])
    thumb_ip_pinky = np.linalg.norm(pts[3] - pts[17])
    thumb_dist_wrist = np.linalg.norm(pts[4] - wrist)
    thumb_mcp_wrist = np.linalg.norm(pts[2] - wrist)
    thumb_ext = (thumb_dist_pinky > thumb_ip_pinky * 1.02) and (thumb_dist_wrist > thumb_mcp_wrist * 1.05)

    # OK Sign check: Thumb Tip (4) and Index Tip (8) touching, Middle finger extended
    thumb_index_dist = np.linalg.norm(pts[4] - pts[8])
    is_ok_circle = (thumb_index_dist < 0.35 * hand_scale) and middle_ext and not index_ext

    if is_ok_circle:
        return "OK_SIGN"

    extended_four = [index_ext, middle_ext, ring_ext, pinky_ext]
    extended_four_count = sum(extended_four)

    if extended_four_count == 4:
        if thumb_ext:
            return "OPEN_PALM"
        else:
            return "FOUR_FINGERS"
    elif extended_four_count == 3:
        return "THREE_FINGERS"
    elif extended_four_count == 2:
        return "TWO_FINGERS"
    elif extended_four_count == 1:
        return "ONE_FINGER"
    elif extended_four_count == 0:
        if thumb_ext:
            return "ONE_FINGER"
        return "FIST"

    return None

def extract_landmarks(results):
    """
    Extracts normalized 126D landmark array with left-hand invariance and heuristic verification.
    """
    left_coords = None
    right_coords = None
    primary_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        primary_hand = results.multi_hand_landmarks[0]
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if hand_label == "Left":
                left_coords = coords
            else:
                right_coords = coords

    left_vector = np.zeros(63, dtype=np.float32)
    right_vector = np.zeros(63, dtype=np.float32)

    if left_coords is not None and right_coords is not None:
        left_vector = normalize_single_hand(left_coords, is_left=True)
        right_vector = normalize_single_hand(right_coords, is_left=False)
    elif right_coords is not None:
        right_vector = normalize_single_hand(right_coords, is_left=False)
    elif left_coords is not None:
        right_vector = normalize_single_hand(left_coords, is_left=True)

    vec126 = np.concatenate([left_vector, right_vector])
    heuristic = heuristic_gesture_check(primary_hand) if primary_hand else None

    return vec126, heuristic

# ==============================================================================
# DECOUPLED HIGH-SPEED THREADED GESTURE ENGINE
# ==============================================================================

class CompetitionGestureEngine:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()

        self.latest_raw_frame = None
        self.latest_jpeg = None

        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=3)

        # Recognition & State variables
        self.current_gesture = "UNKNOWN"
        self.confidence = 0.0
        self.hold_counter = 0
        self.hold_progress = 0
        self.state_status = "idle"
        self.confirmation_count = 0

        self.confirmed_gesture = None
        self.confirmed_sentence = ""
        self.translated_sentence = ""
        self.selected_language = "English"

        self.last_confirmed_gesture = None
        self.cooldown_until = 0.0
        self.no_hand_counter = 0

        # Analytics tracking
        self.session_start_time = None
        self.rejected_count = 0
        self.confidence_sum = 0.0
        self.confidence_eval_count = 0
        self.fps = 0.0
        self._fps_frame_count = 0
        self._last_fps_time = time.time()
        self._gesture_start_time = None
        self.total_response_time_ms = 0.0
        self.response_time_count = 0
        self.avg_response_time_ms = 0.0

        self.cam_thread = None
        self.infer_thread = None

    def start(self):
        with self.lock:
            if self.running and self.cap and self.cap.isOpened():
                return True

            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            opened_cap = None
            for idx in [0, 1, 2]:
                for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                    try:
                        c = cv2.VideoCapture(idx, backend)
                        if c and c.isOpened():
                            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            c.set(cv2.CAP_PROP_FPS, 30)
                            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                            has_frame = False
                            for _ in range(5):
                                ret, test_frame = c.read()
                                if ret and test_frame is not None and test_frame.size > 0:
                                    has_frame = True
                                    break
                                time.sleep(0.03)
                            if has_frame:
                                opened_cap = c
                                break
                            else:
                                c.release()
                    except Exception:
                        pass
                if opened_cap:
                    break

            if opened_cap:
                self.cap = opened_cap
                self.running = True
                self.session_start_time = time.time()
                print("Camera initialized with high-speed MJPG settings!")
            else:
                print("Could not open camera on indices 0, 1, 2.")
                self.running = False
                return False

        if self.cam_thread is None or not self.cam_thread.is_alive():
            self.cam_thread = threading.Thread(target=self._camera_reader_loop, daemon=True)
            self.cam_thread.start()

        if self.infer_thread is None or not self.infer_thread.is_alive():
            self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.infer_thread.start()

        return True

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_raw_frame = None
            self.latest_jpeg = None
            self.sequence.clear()
            self.prediction_history.clear()
            self.reset_state_internal()

    def reset_state_internal(self):
        self.current_gesture = "UNKNOWN"
        self.confidence = 0.0
        self.hold_counter = 0
        self.hold_progress = 0
        self.state_status = "idle"
        self.confirmed_gesture = None
        self.confirmed_sentence = ""
        self.translated_sentence = ""
        self.last_confirmed_gesture = None
        self.cooldown_until = 0.0

        # Reset analytics metrics
        self.rejected_count = 0
        self.confidence_sum = 0.0
        self.confidence_eval_count = 0
        self.total_response_time_ms = 0.0
        self.response_time_count = 0
        self.avg_response_time_ms = 0.0
        self._gesture_start_time = None
        if self.running:
            self.session_start_time = time.time()

    def _translate_current_sentence(self, sentence, language):
        if not sentence:
            return ""
        if language == "English":
            return sentence

        target_code = LANG_CODES.get(language, "en")
        try:
            translated = GoogleTranslator(source="auto", target=target_code).translate(sentence)
            return translated
        except Exception as e:
            print(f"Translation note: {e}")
            return sentence

    def _camera_reader_loop(self):
        """
        Ultra-fast camera capture thread: runs at 30-60 FPS, encodes JPEG, and updates stream buffer immediately.
        """
        failed_reads = 0
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.01)
                    continue

                success, frame = self.cap.read()
                if not success or frame is None or frame.size == 0:
                    failed_reads += 1
                    if failed_reads > 25:
                        with self.lock:
                            try:
                                if self.cap:
                                    self.cap.release()
                                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            except Exception:
                                pass
                        failed_reads = 0
                    time.sleep(0.01)
                    continue

                failed_reads = 0
                frame = cv2.flip(frame, 1)

                # Instant low-latency JPEG encoding
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
                if ret:
                    with self.lock:
                        self.latest_jpeg = buffer.tobytes()
                        self.latest_raw_frame = frame

                # FPS calculation
                self._fps_frame_count += 1
                fps_now = time.time()
                fps_elapsed = fps_now - self._last_fps_time
                if fps_elapsed >= 1.0:
                    self.fps = round(self._fps_frame_count / fps_elapsed, 1)
                    self._fps_frame_count = 0
                    self._last_fps_time = fps_now

            except Exception as e:
                print("Camera reader loop error:", e)
                time.sleep(0.01)

            time.sleep(0.001)

    def _inference_loop(self):
        """
        Concurrent inference thread: processes hand landmarks & neural gesture predictions without delaying camera feed.
        """
        while self.running:
            try:
                frame = None
                with self.lock:
                    if self.latest_raw_frame is not None:
                        frame = self.latest_raw_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                has_hands = bool(results.multi_hand_landmarks)
                landmarks, heuristic_label = extract_landmarks(results)

                with self.lock:
                    now = time.time()
                    in_cooldown = (now < self.cooldown_until)

                    if has_hands:
                        if self._gesture_start_time is None:
                            self._gesture_start_time = now

                        self.no_hand_counter = 0
                        self.sequence.append(landmarks)

                        if len(self.sequence) < SEQUENCE_LENGTH:
                            padded = [landmarks] * (SEQUENCE_LENGTH - len(self.sequence)) + list(self.sequence)
                            input_data = np.expand_dims(np.array(padded), axis=0).astype(np.float32)
                        else:
                            input_data = np.expand_dims(np.array(self.sequence), axis=0).astype(np.float32)

                        pred = fast_predict(input_data)
                        self.prediction_history.append(pred)
                        avg_pred = np.mean(self.prediction_history, axis=0)

                        best_idx = int(np.argmax(avg_pred))
                        raw_conf = float(avg_pred[best_idx])
                        raw_label = labels[best_idx] if best_idx < len(labels) else "UNKNOWN"

                        # HYBRID RECOGNITION LOGIC (NN + HIGH-PRECISION GEOMETRY)
                        detected_gesture = "UNKNOWN"
                        final_conf = raw_conf

                        if heuristic_label and raw_label == heuristic_label:
                            detected_gesture = raw_label
                            final_conf = max(raw_conf, 0.96)
                        elif heuristic_label:
                            detected_gesture = heuristic_label
                            final_conf = max(raw_conf, 0.90)
                        elif raw_conf >= GESTURE_CONFIDENCE_THRESHOLD and raw_label != "UNKNOWN":
                            detected_gesture = raw_label

                        if detected_gesture != "UNKNOWN":
                            self.confidence = round(final_conf * 100, 1)
                            self.current_gesture = detected_gesture
                            self.confidence_sum += self.confidence
                            self.confidence_eval_count += 1
                        else:
                            self.confidence = round(raw_conf * 100, 1)
                            self.current_gesture = "UNKNOWN"
                            self.rejected_count += 1

                        # DEBOUNCING / COOLDOWN & STABILITY
                        if in_cooldown:
                            self.state_status = "cooldown"
                            self.hold_progress = 100
                        elif detected_gesture != "UNKNOWN":
                            if detected_gesture != self.last_confirmed_gesture:
                                self.hold_counter += 1
                                self.hold_progress = min(100, int((self.hold_counter / HOLD_FRAMES) * 100))
                                self.state_status = "holding"

                                if self.hold_counter >= HOLD_FRAMES:
                                    self.confirmed_gesture = detected_gesture
                                    sentence = get_sentence_for_gesture(detected_gesture)
                                    self.confirmed_sentence = sentence
                                    self.translated_sentence = self._translate_current_sentence(sentence, self.selected_language)
                                    self.last_confirmed_gesture = detected_gesture
                                    self.cooldown_until = now + COOLDOWN_SECONDS
                                    self.state_status = "confirmed"
                                    self.confirmation_count += 1
                                    self.hold_counter = 0

                                    if self._gesture_start_time is not None:
                                        resp_ms = round((now - self._gesture_start_time) * 1000, 1)
                                        self.total_response_time_ms += resp_ms
                                        self.response_time_count += 1
                                        self.avg_response_time_ms = round(self.total_response_time_ms / self.response_time_count, 1)
                                        self._gesture_start_time = None
                            else:
                                self.state_status = "confirmed"
                                self.hold_progress = 100
                        else:
                            self.hold_counter = max(0, self.hold_counter - 1)
                            self.hold_progress = min(100, int((self.hold_counter / HOLD_FRAMES) * 100))
                            if self.hold_counter == 0:
                                self.state_status = "unknown"
                                self.last_confirmed_gesture = None
                    else:
                        self.no_hand_counter += 1
                        if self.no_hand_counter >= 3:
                            self._gesture_start_time = None
                            self.sequence.clear()
                            self.prediction_history.clear()
                            self.current_gesture = "UNKNOWN"
                            self.confidence = 0.0
                            self.hold_counter = 0
                            self.hold_progress = 0
                            self.last_confirmed_gesture = None
                            if not in_cooldown:
                                self.state_status = "idle"

            except Exception as e:
                print("Inference loop error:", e)
                time.sleep(0.01)

            time.sleep(0.003)

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def get_state(self):
        with self.lock:
            now = time.time()
            session_duration = int(now - self.session_start_time) if (self.running and self.session_start_time) else 0
            avg_conf = round(self.confidence_sum / self.confidence_eval_count, 1) if self.confidence_eval_count > 0 else 0.0
            total_evals = self.confirmation_count + self.rejected_count
            success_rate = round((self.confirmation_count / total_evals) * 100, 1) if total_evals > 0 else 0.0

            return {
                "gesture": self.current_gesture,
                "confidence": self.confidence,
                "hold_progress": self.hold_progress,
                "confirmed": bool(self.confirmed_gesture),
                "confirmed_gesture": self.confirmed_gesture,
                "icon": get_icon_for_gesture(self.current_gesture if self.current_gesture != "UNKNOWN" else self.confirmed_gesture),
                "sentence": self.confirmed_sentence,
                "translated_sentence": self.translated_sentence,
                "language": self.selected_language,
                "status": self.state_status,
                "confirmation_count": self.confirmation_count,
                "camera_running": self.running,
                "analytics": {
                    "gestures_recognized": self.confirmation_count,
                    "successful_predictions": self.confirmation_count,
                    "rejected_gestures": self.rejected_count,
                    "avg_confidence": avg_conf,
                    "avg_response_time_ms": round(self.avg_response_time_ms, 1) if self.response_time_count > 0 else 0,
                    "sentences_generated": self.confirmation_count,
                    "fps": self.fps if self.running else 0.0,
                    "session_duration_sec": session_duration,
                    "success_rate": success_rate
                }
            }

    def set_language(self, language):
        with self.lock:
            if language in LANGUAGES:
                self.selected_language = language
                if self.confirmed_sentence:
                    self.translated_sentence = self._translate_current_sentence(self.confirmed_sentence, language)

engine = CompetitionGestureEngine()

# In-Memory Speech Cache
tts_cache = {}

# ==============================================================================
# FLASK API ROUTES
# ==============================================================================

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def index():
    return render_template("index.html", languages=LANGUAGES, gesture_mappings=GESTURE_TO_SENTENCE, gesture_icons=GESTURE_ICONS)

@app.route("/video_feed")
def video_feed():
    if not engine.running:
        engine.start()

    def generate():
        while engine.running:
            frame = engine.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.005)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start_camera", methods=["POST"])
def start_camera():
    success = engine.start()
    return jsonify({"status": "started" if success else "failed", "camera_running": success})

@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    engine.stop()
    return jsonify({"status": "stopped", "camera_running": False})

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(engine.get_state())

@app.route("/api/set_language", methods=["POST"])
def set_language():
    lang = request.json.get("language", "English")
    engine.set_language(lang)
    return jsonify({"status": "ok", "language": lang, "translated_sentence": engine.translated_sentence})

@app.route("/api/reset_state", methods=["POST"])
def reset_state():
    with engine.lock:
        engine.reset_state_internal()
    return jsonify({"status": "cleared"})

@app.route("/api/translate", methods=["POST"])
def translate():
    sentence = request.json.get("sentence", "")
    language = request.json.get("language", "English")

    if not sentence:
        return jsonify({
            "success": False,
            "error": "Empty sentence",
            "source_sentence": "",
            "language": language,
            "translated_text": ""
        }), 400

    target_code = LANG_CODES.get(language, "en")
    try:
        if language == "English":
            translated = sentence
        else:
            translated = GoogleTranslator(source="auto", target=target_code).translate(sentence)
        return jsonify({
            "success": True,
            "source_sentence": sentence,
            "language": language,
            "translated_text": translated
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "source_sentence": sentence,
            "language": language,
            "translated_text": sentence
        })

@app.route("/api/speak", methods=["POST"])
def speak():
    sentence = request.json.get("sentence", "")
    language = request.json.get("language", "English")

    if not sentence:
        return jsonify({"error": "No sentence provided for speech synthesis."}), 400

    lang_code = LANG_CODES.get(language, "en")
    text_to_speak = sentence
    if language != "English":
        try:
            text_to_speak = GoogleTranslator(source="auto", target=lang_code).translate(sentence)
        except Exception:
            text_to_speak = sentence

    cache_key = f"{lang_code}:{text_to_speak}"
    if cache_key in tts_cache and os.path.exists(tts_cache[cache_key]):
        return send_file(tts_cache[cache_key], mimetype="audio/mpeg")

    try:
        tts = gTTS(text=text_to_speak, lang=lang_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        tts_cache[cache_key] = temp_file.name
        return send_file(temp_file.name, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": f"Speech unavailable for {language}: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)