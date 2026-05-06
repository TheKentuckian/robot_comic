"""MediaPipe head tracker backed by reachy_mini_toolbox."""

import os

import numpy as np
from numpy.typing import NDArray

from robot_comic.vision.head_tracking import HeadTracker, HeadTrackerResult


class MediapipeHeadTracker:
    """MediaPipe head tracker provided by reachy_mini_toolbox."""

    def __init__(self) -> None:
        """Initialize the toolbox head tracker lazily."""
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/robot-comic-matplotlib")

        from reachy_mini_toolbox import vision

        self._tracker: HeadTracker = vision.HeadTracker()

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.ascontiguousarray(img[..., ::-1])
        return self._tracker.get_head_position(img)
