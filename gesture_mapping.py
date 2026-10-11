"""
Gesture Mapping Utility Module for GestureTalk Competition Edition.
Provides canonical sentence lookups, icon retrievals, and label mappings.
"""

from config import GESTURE_TO_SENTENCE, GESTURE_ICONS, GESTURE_CLASSES

def get_sentence_for_gesture(gesture_name: str) -> str:
    """
    Returns the canonical English sentence for a given recognized gesture name.
    Returns empty string if gesture is UNKNOWN or invalid.
    """
    if not gesture_name or gesture_name == "UNKNOWN":
        return ""
    return GESTURE_TO_SENTENCE.get(gesture_name, "")

def get_icon_for_gesture(gesture_name: str) -> str:
    """
    Returns the UI display icon/emoji for a given gesture name.
    """
    return GESTURE_ICONS.get(gesture_name, "❓")

def is_valid_gesture(gesture_name: str) -> bool:
    """
    Checks if gesture is one of the 6 valid competition trigger classes.
    """
    return gesture_name in GESTURE_CLASSES

def get_all_mappings():
    """
    Returns complete dictionary mapping gesture names to sentences and icons.
    """
    return {
        cls: {
            "sentence": GESTURE_TO_SENTENCE[cls],
            "icon": GESTURE_ICONS[cls]
        }
        for cls in GESTURE_CLASSES
    }
