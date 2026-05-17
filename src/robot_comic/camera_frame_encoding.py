"""Helpers for encoding camera frames."""

from fractions import Fraction

import av
import numpy as np
from numpy.typing import NDArray


def encode_bgr_frame_as_png(frame: NDArray[np.uint8]) -> bytes:
    """Encode a BGR camera frame as lossless PNG bytes.

    Uses PyAV (already a core dependency) so no additional libraries
    are required.  The frame is converted BGR → RGB before encoding
    because ``av.VideoFrame.from_ndarray`` with ``format="rgb24"``
    expects channel order R, G, B.
    """
    rgb_frame = np.ascontiguousarray(frame[..., ::-1])
    video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

    codec = av.CodecContext.create("png", "w")
    codec.width = rgb_frame.shape[1]
    codec.height = rgb_frame.shape[0]
    codec.pix_fmt = "rgb24"

    packets = codec.encode(video_frame)
    packets += codec.encode(None)
    if not packets:
        raise RuntimeError("Failed to encode frame as PNG")

    return b"".join(bytes(packet) for packet in packets)


def encode_bgr_frame_as_jpeg(frame: NDArray[np.uint8]) -> bytes:
    """Encode a BGR camera frame as JPEG bytes."""
    rgb_frame = np.ascontiguousarray(frame[..., ::-1])
    video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = rgb_frame.shape[1]  # type: ignore[attr-defined]
    codec.height = rgb_frame.shape[0]  # type: ignore[attr-defined]
    codec.pix_fmt = "yuvj444p"  # type: ignore[attr-defined]
    codec.time_base = Fraction(1, 1)
    codec.options = {"qscale": "3"}

    packets = codec.encode(video_frame)  # type: ignore[attr-defined]
    packets += codec.encode(None)  # type: ignore[attr-defined]
    if not packets:
        raise RuntimeError("Failed to encode frame as JPEG")

    return b"".join(bytes(packet) for packet in packets)
