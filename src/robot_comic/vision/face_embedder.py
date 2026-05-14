"""Face-embedder interface and stub implementation.

Defines the ``FaceEmbedder`` Protocol that any concrete embedding backend must
satisfy.  A ``StubFaceEmbedder`` is provided for development and testing; it
always returns ``None`` (no real embeddings produced) until a real backend such
as ``face_recognition`` / ``dlib`` or a small ONNX model is wired in a
follow-up PR.

Enable the face-recognition pipeline with the ``REACHY_MINI_FACE_RECOGNITION_ENABLED``
env var (default ``False``).  Until a real embedder is available the flag should
remain off.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FaceEmbedder(Protocol):
    """Protocol that every face-embedding backend must satisfy.

    Implementors receive a BGR or RGB video frame as a NumPy array and return
    a 1-D float embedding suitable for cosine-similarity comparison, or
    ``None`` when no face is detected in the frame.
    """

    def embed(self, frame: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Extract a face embedding from *frame*.

        Args:
            frame: A single video frame as a NumPy array with shape
                ``(H, W, C)`` in BGR or RGB order and dtype convertible to
                float32.

        Returns:
            A 1-D NumPy array of float32 values representing the face
            embedding, or ``None`` if no face could be detected.

        """
        ...


class StubFaceEmbedder:
    """No-op embedder used until a real backend is implemented.

    ``embed`` always returns ``None``; it never produces real embeddings and
    therefore never triggers the face-recognition path.  Replace this class
    with a concrete implementation (e.g. ``InsightFaceEmbedder``) in a
    follow-up PR.
    """

    def embed(self, frame: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Return ``None`` unconditionally — no real embedding produced."""
        return None
