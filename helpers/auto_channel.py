import numpy as np
from PyQt5.QtWidgets import QMessageBox


def detect_best_channel(patch: np.ndarray) -> str:
    """
    Analyze the given image patch (H x W x 3 BGR) and return the channel ('B','G','R')
    with the highest variance (contrast), as a heuristic for best signal.

    Args:
        patch: A NumPy array of shape (H, W, 3) in BGR order.

    Returns:
        The channel identifier with highest variance: 'B', 'G', or 'R'.
    """
    QMessageBox.information(None, "Auto-Channel Selection", "Auto-Channel Selection is currently under development.")
    return 'R'  # Default to 'R' for now
