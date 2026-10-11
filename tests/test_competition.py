import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GESTURE_TO_SENTENCE, GESTURE_CLASSES, GESTURE_CONFIDENCE_THRESHOLD
from gesture_mapping import get_sentence_for_gesture, get_icon_for_gesture, is_valid_gesture

class TestGestureTalkCompetition(unittest.TestCase):

    def test_gesture_sentence_mapping(self):
        """Verify exact canonical sentence mapping for all 6 competition gestures."""
        expected_mappings = {
            "ONE_FINGER": "I need your help.",
            "TWO_FINGERS": "I want some water.",
            "THREE_FINGERS": "Where is the bathroom?",
            "OPEN_PALM": "Hello, how are you?",
            "OK_SIGN": "Thank you very much.",
            "FIST": "I am going home."
        }

        for gesture, expected_sentence in expected_mappings.items():
            self.assertEqual(GESTURE_TO_SENTENCE[gesture], expected_sentence)
            self.assertEqual(get_sentence_for_gesture(gesture), expected_sentence)

    def test_gesture_validation(self):
        """Verify valid competition gesture recognition and rejection of unknown gestures."""
        for gesture in GESTURE_CLASSES:
            self.assertTrue(is_valid_gesture(gesture))

        self.assertFalse(is_valid_gesture("INVALID_GESTURE"))
        self.assertFalse(is_valid_gesture("HELLO")) # Old 28-class label
        self.assertEqual(get_sentence_for_gesture("UNKNOWN"), "")

    def test_confidence_threshold_setting(self):
        """Verify confidence threshold is set to 0.80."""
        self.assertEqual(GESTURE_CONFIDENCE_THRESHOLD, 0.80)

    def test_icons_retrieval(self):
        """Verify emoji/icon mapping returns non-empty icon strings."""
        for gesture in GESTURE_CLASSES:
            icon = get_icon_for_gesture(gesture)
            self.assertTrue(len(icon) > 0)
            self.assertNotEqual(icon, "❓")

if __name__ == "__main__":
    unittest.main()
