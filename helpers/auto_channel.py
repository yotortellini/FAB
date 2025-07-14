import numpy as np


def detect_best_channel(patch: np.ndarray) -> str:
    """
    Analyze the given image patch (H x W x 3 BGR) and return the channel ('B','G','R')
    with the highest variance (contrast), as a heuristic for best signal.

    Args:
        patch: A NumPy array of shape (H, W, 3) in BGR order.

    Returns:
        The channel identifier with highest variance: 'B', 'G', or 'R'.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("patch must be HxWx3 in BGR format")

    # Compute variance per channel
    variances = patch.reshape(-1, 3).var(axis=0)
    # OpenCV default is BGR order
    channel_idx = int(np.argmax(variances))
    return ['B', 'G', 'R'][channel_idx]
