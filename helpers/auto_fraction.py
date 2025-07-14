from typing import Tuple
import numpy as np

from models.video_session import VideoSession
from engine.video_engine import VideoEngine


def estimate_fractions(session: VideoSession, rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Estimate reasonable start and end fractions for a given ROI based on intensity change.
    Currently returns full-range [0.0, 1.0] as placeholder.
    TODO: Sample the ROI over the clip to detect rising/falling edges.

    Args:
        session: The VideoSession containing start_frame, end_frame, and fps.
        rect: (x, y, w, h) ROI in deskewed coordinates.

    Returns:
        A tuple (start_frac, end_frac) between 0.0 and 1.0.
    """
    # Placeholder implementation: full range
    return 0.0, 1.0
