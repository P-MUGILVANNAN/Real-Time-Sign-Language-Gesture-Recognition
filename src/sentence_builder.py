"""
Sentence and gesture sequence builder module.
Accumulates stable gesture predictions into coherent phrases,
prevents duplicate triggers during sustained signing, and provides editing controls.
"""

import datetime
from typing import List, Optional
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("SentenceBuilder")


class GestureSentenceBuilder:
    """
    Accumulates recognized dynamic signs into an ordered sentence sequence.
    """
    def __init__(
        self,
        neutral_label: str = "No confident gesture detected",
        cooldown_frames: int = 15
    ):
        self.neutral_label = neutral_label
        self.cooldown_frames = cooldown_frames

        self.tokens: List[str] = []
        self.token_timestamps: List[str] = []

        self.last_added_gesture: Optional[str] = None
        self.cooldown_counter: int = 0

    def process_gesture_event(self, current_gesture: str, is_new_trigger: bool):
        """
        Receives an updated prediction from the smoother.
        Inserts token when a genuine new gesture occurs, respecting cooldown.
        """
        # If user returns to neutral, clear the repetition lock
        if current_gesture == self.neutral_label:
            self.last_added_gesture = None
            self.cooldown_counter = 0
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Only insert if it is a new trigger and not a duplicate sustained sign without pause
        if is_new_trigger and (current_gesture != self.last_added_gesture or self.cooldown_counter == 0):
            self.tokens.append(current_gesture)
            self.token_timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))
            self.last_added_gesture = current_gesture
            self.cooldown_counter = self.cooldown_frames
            logger.info(f"Appended token: '{current_gesture}'. Current sequence: {self.get_sentence()}")

    def get_sentence(self) -> str:
        """Returns readable capitalized sequence string."""
        if not self.tokens:
            return ""
        return " ".join([t.replace("_", " ").capitalize() for t in self.tokens])

    def undo_last_token(self) -> Optional[str]:
        """Removes the most recent gesture token."""
        if self.tokens:
            removed = self.tokens.pop()
            self.token_timestamps.pop()
            self.last_added_gesture = None
            return removed
        return None

    def clear(self):
        """Clears all tokens in the sentence buffer."""
        self.tokens.clear()
        self.token_timestamps.clear()
        self.last_added_gesture = None
        self.cooldown_counter = 0

    def export_csv(self, filepath: str):
        """Exports the token sequence and timestamps to CSV."""
        df = pd.DataFrame({
            "timestamp": self.token_timestamps,
            "gesture_token": self.tokens
        })
        df.to_csv(filepath, index=False)
        logger.info(f"Exported sentence tokens to {filepath}")
