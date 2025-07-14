import numpy as np
from typing import Tuple

from models.video_session import VideoSession
from engine.video_engine import VideoEngine


def sample_background_profile(
    session: VideoSession,
    engine: VideoEngine,
    rect: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Sample the mean intensity in the specified blank region over each frame in the session's range.

    Args:
        session: VideoSession containing start_frame, end_frame, fps.
        engine: VideoEngine to fetch frames.
        rect: (x, y, w, h) region assumed to be blank background.

    Returns:
        1D array of background intensities per frame index from start_frame to end_frame.
    """
    x, y, w, h = rect
    start, end = session.start_frame, session.end_frame
    total = end - start
    profile = np.zeros(total, dtype=float)
    for i, frame_idx in enumerate(range(start, end)):
        frame = engine.get_frame(frame_idx)
        gray = frame[y:y+h, x:x+w]
        profile[i] = float(np.mean(gray))
    return profile


def apply_lighting_correction(
    frame: np.ndarray,
    background_profile: np.ndarray,
    frame_index: int,
    method: str = 'subtract'
) -> np.ndarray:
    """
    Apply lighting correction to a single frame based on a precomputed background profile.

    Args:
        frame: BGR image array for the frame.
        background_profile: 1D array of background intensities per frame.
        frame_index: global frame index (>= session.start_frame).
        method: correction method, currently only 'subtract' supported.

    Returns:
        Corrected BGR image array.
    """
    if method != 'subtract':
        raise ValueError("Unsupported correction method: {method}")
    # Determine index in profile
    idx = frame_index - session.start_frame
    bg = background_profile[idx]
    # Convert to float to avoid underflow
    corrected = frame.astype(float) - bg
    # Clip and convert back to uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected
