import cv2
import numpy as np
from typing import Tuple, Dict

class VideoEngine:
    """
    Wraps OpenCV VideoCapture and provides methods for loading videos,
    retrieving frames, applying deskew rotations, and cropping.
    """
    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.video_path: str = ""

    def load_video(self, path: str) -> Dict[str, float]:
        """
        Open the video file at the given path and return basic metadata.
        Raises:
            IOError: If the video cannot be opened.

        Returns:
            A dict with keys:
                - 'fps': frames per second of the video
                - 'frame_count': total number of frames in the video
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file at {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap = cap
        self.video_path = path
        return {'fps': fps, 'frame_count': frame_count}

    def get_frame(self, index: int) -> np.ndarray:
        """
        Retrieve a single frame by its index.

        Raises:
            RuntimeError: If load_video hasn't been called.
            IOError: If the frame cannot be read.

        Returns:
            The BGR image array for the requested frame.
        """
        if self.cap is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"Failed to read frame at index {index}")
        return frame

    def apply_rotation(self, frame: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Deskew a frame using the provided affine rotation matrix.

        Args:
            frame: Source image array.
            rotation_matrix: 2x3 affine transform matrix.

        Returns:
            The warped (rotated) image.
        """
        h, w = frame.shape[:2]
        rotated = cv2.warpAffine(frame, rotation_matrix, (w, h))
        return rotated

    def crop_frame(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop a rectangular region from the frame.

        Args:
            frame: Source image array.
            rect: (x, y, width, height) rectangle to crop.

        Returns:
            The cropped sub-image.
        """
        x, y, w, h = rect
        return frame[y:y+h, x:x+w]

    def release(self):
        """
        Release any held video resources.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
