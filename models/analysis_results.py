from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

@dataclass
class AnalysisResults:
    """
    Holds the results of volume and flow analysis for a session.

    Attributes:
        time: 1D array of time points corresponding to measurements.
        volume: 1D array of volume values over time.
        flow: 1D array of instantaneous flow values (first derivative of volume).
        fill_flow: Optional 1D array of flow during filling phase.
        drain_flow: Optional 1D array of flow during drainage phase.
        smoothing_window: Window size used for any smoothing applied.
        plot_paths: Optional dict mapping plot names (e.g., 'volume', 'flow') to file paths of exported images.
    """
    time: np.ndarray
    volume: np.ndarray
    flow: np.ndarray
    fill_flow: Optional[np.ndarray] = None
    drain_flow: Optional[np.ndarray] = None
    smoothing_window: int = 0
    plot_paths: Dict[str, str] = field(default_factory=dict)

    def add_plot_path(self, name: str, path: str) -> None:
        """
        Record the file path for a generated plot image.
        """
        self.plot_paths[name] = path

    def get_plot_path(self, name: str) -> Optional[str]:
        """
        Retrieve the file path for a given plot name, if available.
        """
        return self.plot_paths.get(name)
