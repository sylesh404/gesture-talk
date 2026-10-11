"""
GestureTalk - Competition Edition Configuration
Centralized configuration parameters, thresholds, and predefined gesture-to-sentence mappings.
"""

import os

# ==============================================================================
# INFERENCE & STABILITY THRESHOLDS
# ==============================================================================
GESTURE_CONFIDENCE_THRESHOLD = 0.80  # Minimum probability score required to accept gesture
HOLD_FRAMES = 2                       # Consecutive required hold frames (~0.05 sec for instant response)
COOLDOWN_SECONDS = 0.5               # Fast debounce cooldown period after confirmation
SEQUENCE_LENGTH = 20                 # Number of landmark frames per temporal sequence
NUM_LANDMARK_COORDINATES = 126       # 21 3D landmarks per hand x 2 hands x 3 (x,y,z) = 126

# ==============================================================================
# COMPETITION GESTURE CLASSES (ORDERED)
# ==============================================================================
GESTURE_CLASSES = [
    "ONE_FINGER",
    "TWO_FINGERS",
    "THREE_FINGERS",
    "FOUR_FINGERS",
    "OPEN_PALM",
    "OK_SIGN",
    "FIST"
]

# ==============================================================================
# CANONICAL GESTURE-TO-SENTENCE MAPPING
# ==============================================================================
GESTURE_TO_SENTENCE = {
    "ONE_FINGER": "I need your help.",
    "TWO_FINGERS": "I want some water.",
    "THREE_FINGERS": "Where is the bathroom?",
    "FOUR_FINGERS": "Please bring me food.",
    "OPEN_PALM": "Hello, how are you?",
    "OK_SIGN": "Thank you very much.",
    "FIST": "I am going home."
}

# ==============================================================================
# GESTURE DISPLAY EMOJIS / ICONS
# ==============================================================================
GESTURE_ICONS = {
    "ONE_FINGER": "☝️",
    "TWO_FINGERS": "✌️",
    "THREE_FINGERS": "🤟",
    "FOUR_FINGERS": "🖖",
    "OPEN_PALM": "✋",
    "OK_SIGN": "👌",
    "FIST": "✊",
    "UNKNOWN": "❓"
}

# ==============================================================================
# FILE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_KERAS = os.path.join(BASE_DIR, "models", "gesture_model.keras")
MODEL_PATH_TFLITE = os.path.join(BASE_DIR, "models", "gesture_model.tflite")
LABELS_PATH_CSV = os.path.join(BASE_DIR, "labels", "gesture_labels.csv")
LABELS_PATH_NPY = os.path.join(BASE_DIR, "labels", "gesture_labels.npy")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# Legacy backup paths
LEGACY_DIR = os.path.join(BASE_DIR, "legacy_model")

# Supported translation languages
LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Nepali": "ne",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Thai": "th",
    "Russian": "ru",
    "Portuguese": "pt"
}

LANGUAGES = list(LANG_CODES.keys())
