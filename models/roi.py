from dataclasses import dataclass
from typing import Tuple

@dataclass
class ROI:
    """
    Region of interest definition for analysis.

    Attributes:
        name: Unique identifier for the ROI.
        rect: (x, y, width, height) on the deskewed frame.
        channel: 'R', 'G', 'B', or 'auto'.
        total_volume: Expected or placeholder total volume value.
        start_frac: Fraction of clip start (0.0 to 1.0).
        end_frac: Fraction of clip end (0.0 to 1.0).
        interval: Sampling interval in frames (> 0).
        smoothing_window: Window size for smoothing (> 0).
    """
    name: str
    rect: Tuple[int, int, int, int]
    channel: str
    total_volume: float
    start_frac: float
    end_frac: float
    interval: int
    smoothing_window: int

    def __post_init__(self):
        # Validate fractions
        if not (0.0 <= self.start_frac <= 1.0):
            raise ValueError("start_frac must be between 0 and 1")
        if not (0.0 <= self.end_frac <= 1.0):
            raise ValueError("end_frac must be between 0 and 1")
        if self.start_frac >= self.end_frac:
            raise ValueError("start_frac must be less than end_frac")
        # Validate sampling
        if self.interval <= 0:
            raise ValueError("interval must be > 0")
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be > 0")
        # Validate channel
        valid_channels = ('R', 'G', 'B', 'auto')
        if self.channel not in valid_channels:
            raise ValueError(f"channel must be one of {valid_channels}")
