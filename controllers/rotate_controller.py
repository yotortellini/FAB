# controllers/rotate_controller.py

import cv2
import numpy as np
import math
import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QSlider, QGroupBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from typing import Optional, Callable, Tuple

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class RotateController(QWidget):
    """
    Controller for Step 3: Rotate Video
    Allows rotating video frames by:
    - Drawing a line that should be horizontal
    - Fine-tuning with angle slider
    """

    def __init__(
        self,
        parent: QWidget,
        session: VideoSession,
        engine: VideoEngine,
        on_complete: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)
        
        self.angle = 0
        self.current_frame = None
        self.original_frame = None  # Store the clean original frame
        # Store line endpoints in original image coordinates
        self.line_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.drawing = False
        self.start_pos = None
        self.end_pos = None
        
        self._build_ui()
        # Don't update preview on init - wait for video to be loaded

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 3: Rotate Video")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Draw a line across a feature that should be horizontal, "
            "or use the slider for fine adjustments."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Preview frame
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("border: 1px solid #ccc;")
        # Enable mouse tracking for line drawing
        self.preview.setMouseTracking(True)
        self.preview.mousePressEvent = self._on_mouse_press
        self.preview.mouseMoveEvent = self._on_mouse_move
        self.preview.mouseReleaseEvent = self._on_mouse_release
        layout.addWidget(self.preview)

        # Controls group
        controls_group = QGroupBox("Rotation Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        # Angle controls
        angle_layout = QHBoxLayout()
        
        # Angle spinbox
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(-180, 180)
        self.angle_spin.setSingleStep(1)
        self.angle_spin.valueChanged.connect(self._on_angle_changed)
        angle_layout.addWidget(QLabel("Angle:"))
        angle_layout.addWidget(self.angle_spin)

        # Angle slider
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._on_slider_changed)
        angle_layout.addWidget(self.angle_slider)
        
        controls_layout.addLayout(angle_layout)

        # Reset button
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Rotation")
        self.reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_btn)
        controls_layout.addLayout(button_layout)
        
        layout.addWidget(controls_group)

    def _on_mouse_press(self, event):
        """Handle mouse press events on the preview"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = event.pos()
            self.end_pos = None

    def _on_mouse_move(self, event):
        """Handle mouse move events on the preview"""
        if self.drawing:
            self.end_pos = event.pos()
            self._update_preview()

    def _on_mouse_release(self, event):
        """Handle mouse release events on the preview"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pos = event.pos()
            # Convert drawn widget coordinates to original image coordinates
            if self.original_frame is None:
                self.original_frame = self.engine.get_frame(self.session.start_frame)
                if self.original_frame is None:
                    return
            img_h, img_w = self.original_frame.shape[:2]
            preview_size = self.preview.size()
            scale = min(preview_size.width() / img_w, preview_size.height() / img_h)
            offset_x = (preview_size.width() - img_w * scale) / 2
            offset_y = (preview_size.height() - img_h * scale) / 2
            x0 = int((self.start_pos.x() - offset_x) / scale)
            y0 = int((self.start_pos.y() - offset_y) / scale)
            x1 = int((self.end_pos.x() - offset_x) / scale)
            y1 = int((self.end_pos.y() - offset_y) / scale)
            x0 = max(0, min(img_w - 1, x0))
            y0 = max(0, min(img_h - 1, y0))
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            self.line_points = ((x0, y0), (x1, y1))
            self._calculate_angle()
            self._update_preview()

    def _calculate_angle(self):
        """Calculate rotation angle from the drawn line"""
        if self.line_points:
            (x0, y0), (x1, y1) = self.line_points
            dx = x1 - x0
            dy = y1 - y0
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            self.angle_spin.setValue(int(angle_deg))
            self.angle_slider.setValue(int(angle_deg))
            logging.info(f"Line drawn, angle: {angle_deg:.1f}°")

    def _on_angle_changed(self, value):
        """Handle angle spinbox changes"""
        self.angle_slider.setValue(value)
        self._update_preview()

    def _on_slider_changed(self, value):
        """Handle angle slider changes"""
        self.angle_spin.setValue(value)
        self._update_preview()

    def _on_reset(self):
        """Reset rotation to zero"""
        self.angle_spin.setValue(0)
        self.angle_slider.setValue(0)
        self.start_pos = None
        self.end_pos = None
        # Reset the original frame so a new line can be drawn
        self.original_frame = None
        self.line_points = None
        self._update_preview()

    def _update_preview(self):
        """Update the preview with the current frame and rotation using baked line approach"""
        # Check if video is loaded and session has a start frame
        if (hasattr(self.session, 'start_frame') and 
            self.session.start_frame is not None and 
            hasattr(self.engine, 'cap') and 
            self.engine.cap is not None):
            
            # Get the original frame if we don't have it yet
            if self.original_frame is None:
                self.original_frame = self.engine.get_frame(self.session.start_frame)
                if self.original_frame is None:
                    return
            
            # Start with the original clean frame
            frame = self.original_frame.copy()

            # Apply rotation transform (even if angle is zero)
            current_angle = self.angle_spin.value() if hasattr(self, 'angle_spin') else 0
            # Get original dimensions and compute center
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            # Build rotation matrix
            M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            # Compute new bounds
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            # Adjust translation to keep image centered
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            # Warp original into rotated frame
            frame = cv2.warpAffine(frame, M, (new_w, new_h))
            # Draw line (rotated) if present
            if self.line_points is not None:
                (x0, y0), (x1, y1) = self.line_points
                pt0 = np.dot(M, np.array([x0, y0, 1]))
                pt1 = np.dot(M, np.array([x1, y1, 1]))
                p0 = (int(pt0[0]), int(pt0[1]))
                p1 = (int(pt1[0]), int(pt1[1]))
                cv2.line(frame, p0, p1, (0, 0, 255), 3)
                cv2.circle(frame, p0, 5, (0, 0, 255), -1)
                cv2.circle(frame, p1, 5, (0, 0, 255), -1)
            
            self.current_frame = frame.copy()
            
            # Convert to RGB for Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage and then QPixmap
            h, w = rgb_frame.shape[:2]
            bytes_per_line = 3 * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale pixmap to fit preview while maintaining aspect ratio
            preview_size = self.preview.size()
            scaled_pixmap = pixmap.scaled(preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set the pixmap - no need for additional drawing since line is baked in
            self.preview.setPixmap(scaled_pixmap)

    def initialize_step(self):
        """Initialize the step when it becomes active"""
        self._update_preview()

    def save_rotation(self):
        """Save the current rotation angle to the session"""
        current_angle = self.angle_spin.value() if hasattr(self, 'angle_spin') else 0
        if current_angle != 0:
            if self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
                self.session.rotation_matrix = M
        return True  # Always return True since rotation is optional
