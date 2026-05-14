"""Head-tracking backends and process helpers."""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, TypeAlias, SupportsFloat


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


HeadTrackerResult: TypeAlias = "tuple[NDArray[np.float32] | None, SupportsFloat | None]"


class HeadTracker(Protocol):
    """Shared interface for optional head-tracking backends."""

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""
