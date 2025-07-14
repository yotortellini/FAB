from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

# forward references to be filled when other modules are created
from models.roi import ROI
from models.analysis_results import AnalysisResults

@dataclass
class VideoSession:
    """
    Encapsulates the state of a video processing session, including loaded video metadata,
    temporal clipping, rotation transform, defined ROIs, and analysis results.
    """
    path: str = ''
    fps: float = 0.0
    frame_count: int = 0
    start_frame: int = 0
    end_frame: int = 0
    time_multiplier: float = 1.0
    rotation_matrix: Optional[np.ndarray] = None
    rois: Dict[str, ROI] = field(default_factory=dict)
    analysis_results: Optional[AnalysisResults] = None

    def validate_frame_range(self):
        """
        Ensure the start/end frames are within the video and start < end.
        """
        if self.start_frame < 0 or self.end_frame > self.frame_count:
            raise ValueError(
                f"Frame range [{self.start_frame}, {self.end_frame}] out of bounds (0, {self.frame_count})"
            )
        if self.start_frame >= self.end_frame:
            raise ValueError("start_frame must be less than end_frame")

    def set_frame_range(self, start: int, end: int):
        """
        Set and validate the temporal clipping range.
        """
        self.start_frame = start
        self.end_frame = end
        self.validate_frame_range()

    def add_roi(self, roi: ROI):
        """
        Add an ROI to the session, ensuring unique names.
        """
        if roi.name in self.rois:
            raise KeyError(f"ROI named '{roi.name}' already exists")
        self.rois[roi.name] = roi

    def remove_roi(self, name: str):
        """
        Remove an ROI by name.
        """
        if name not in self.rois:
            raise KeyError(f"No ROI named '{name}'")
        del self.rois[name]
