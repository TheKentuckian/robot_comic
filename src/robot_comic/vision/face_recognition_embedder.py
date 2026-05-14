"""Concrete face embedder backed by the ``face_recognition`` library (dlib HOG).

This module provides :class:`FaceRecognitionEmbedder`, a drop-in replacement for
:class:`~robot_comic.vision.face_embedder.StubFaceEmbedder` that produces real
128-D face embeddings suitable for cosine-similarity matching via
:class:`~robot_comic.vision.face_db.FaceDatabase`.

The ``face_recognition`` package is **optional** — install it with::

    pip install -e '.[face_recognition]'

If the library is not installed, constructing :class:`FaceRecognitionEmbedder`
raises :class:`ImportError` with a helpful message.  The application falls back
to :class:`~robot_comic.vision.face_embedder.StubFaceEmbedder` on that error.

Pi-5 latency (HOG mode, 640×480 frame): ~150–250 ms per call.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class FaceRecognitionEmbedder:
    """128-D face embedder powered by the ``face_recognition`` / dlib library.

    Satisfies the :class:`~robot_comic.vision.face_embedder.FaceEmbedder` Protocol.

    Args:
        model: Detection model passed to ``face_recognition.face_locations``.
            ``"hog"`` (default) is CPU-friendly and Pi-5 viable.
            ``"cnn"`` is more accurate but much slower without a GPU.

    Raises:
        ImportError: When the ``face_recognition`` package is not installed.

    """

    def __init__(self, model: str = "hog") -> None:
        """Initialise and verify that ``face_recognition`` is available."""
        try:
            import face_recognition as _fr  # noqa: F401 — optional dep, not in type stubs

            self._fr: Any = _fr
        except ImportError as exc:
            raise ImportError(
                "The 'face_recognition' library is required for FaceRecognitionEmbedder. "
                "Install it with:  pip install -e '.[face_recognition]'"
            ) from exc
        self._model = model

    def embed(self, frame: NDArray[np.float32]) -> NDArray[np.float64] | None:
        """Extract a 128-D face embedding from *frame*.

        Picks the **largest** detected face (closest to the camera) and returns
        its embedding.  Returns ``None`` when no face is found.

        Args:
            frame: A single video frame as a NumPy array with shape ``(H, W, C)``
                in BGR (OpenCV) or RGB order and dtype convertible to uint8.
                ``face_recognition`` expects RGB; BGR frames are converted
                automatically.

        Returns:
            A 1-D ``np.ndarray`` of shape ``(128,)`` and dtype ``float64``, or
            ``None`` when no face is detected.

        """
        # Convert to uint8 — face_recognition requires RGB uint8.
        frame_u8: NDArray[np.uint8]
        if frame.dtype != np.uint8:
            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_u8 = frame.view(np.uint8)

        # Detect face locations (returns list of (top, right, bottom, left) tuples).
        locations = self._fr.face_locations(frame_u8, model=self._model)
        if not locations:
            logger.debug("FaceRecognitionEmbedder: no face detected in frame")
            return None

        # Pick the largest face (biggest bounding-box area) — assumes closest subject.
        def _area(loc: tuple[int, int, int, int]) -> int:
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        largest = max(locations, key=_area)

        # Compute the 128-D encoding for that single location.
        encodings = self._fr.face_encodings(frame_u8, [largest])
        if not encodings:
            logger.debug("FaceRecognitionEmbedder: face_encodings returned empty list")
            return None

        result: NDArray[np.float64] = np.asarray(encodings[0], dtype=np.float64)
        return result
