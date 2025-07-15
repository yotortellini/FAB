# controllers/rotate_controller.py

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QSpinBox, QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from typing import Optional, Callable

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
        self.base_frame = None  # Store the original frame
        self.rotated_frame = None
        
        # Line drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.display_to_image_ratio = 1.0  # Track scaling ratio
        
        self._build_ui()
        self._load_base_frame()
        self._update_preview()

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

        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_btn)

        button_layout.addStretch()

        self.confirm_btn = QPushButton("Confirm Rotation")
        self.confirm_btn.clicked.connect(self._on_confirm)
        button_layout.addWidget(self.confirm_btn)
        
        controls_layout.addLayout(button_layout)
        layout.addWidget(controls_group)

    def _load_base_frame(self):
        """Load and store the base frame"""
        if self.engine.cap:
            self.base_frame = self.engine.get_frame(self.session.start_frame)

    def _get_image_position(self, pos):
        """Convert display coordinates to image coordinates"""
        if not self.preview.pixmap():
            return None

        # Get the geometry of the displayed image
        label_rect = self.preview.rect()
        pixmap = self.preview.pixmap()
        image_rect = pixmap.rect()
        
        # Calculate the actual display rectangle (maintaining aspect ratio)
        display_rect = image_rect
        display_rect.moveCenter(label_rect.center())
        
        # Check if click is within the image area
        if not display_rect.contains(pos):
            return None

        # Calculate scaling ratios
        scale_x = image_rect.width() / display_rect.width()
        scale_y = image_rect.height() / display_rect.height()
        self.display_to_image_ratio = scale_x  # Store for later use

        # Convert to image coordinates
        image_x = (pos.x() - display_rect.left()) * scale_x
        image_y = (pos.y() - display_rect.top()) * scale_y

        return QPoint(int(image_x), int(image_y))

    def _on_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            # Convert to image coordinates
            pos = self._get_image_position(event.pos())
            if pos:
                self.drawing = True
                self.start_point = pos
                self.end_point = pos
                self._update_preview()

    def _on_mouse_move(self, event):
        if self.drawing:
            pos = self._get_image_position(event.pos())
            if pos:
                self.end_point = pos
                self._update_preview()

    def _on_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            pos = self._get_image_position(event.pos())
            if pos:
                self.end_point = pos
                self._calculate_and_apply_angle()

    def _calculate_and_apply_angle(self):
        """Calculate and apply rotation angle from the drawn line"""
        if not (self.start_point and self.end_point):
            return

        # Calculate angle relative to horizontal
        dx = self.end_point.x() - self.start_point.x()
        dy = self.end_point.y() - self.start_point.y()
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Update the angle control
        self.angle_spin.setValue(-angle)  # Negative because we want to rotate to horizontal

    def _update_preview(self):
        if self.base_frame is None:
            return

        # Make a copy of the base frame
        frame = self.base_frame.copy()

        # Apply rotation if needed
        if self.angle != 0:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))

        # Convert to Qt image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Draw the line if we have points
        if self.start_point and self.end_point:
            painter = QPainter(pixmap)
            
            # Set up pen for line drawing
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(3)  # Thicker line
            painter.setPen(pen)

            # Draw the line
            if self.angle != 0:
                # Rotate points with the image
                center = QPoint(w // 2, h // 2)
                angle_rad = np.radians(self.angle)
                rot_matrix = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ])
                
                # Rotate start point
                start_vec = np.array([self.start_point.x() - center.x(),
                                    self.start_point.y() - center.y()])
                rotated_start = np.dot(rot_matrix, start_vec)
                start_point = QPoint(int(rotated_start[0] + center.x()),
                                   int(rotated_start[1] + center.y()))
                
                # Rotate end point
                end_vec = np.array([self.end_point.x() - center.x(),
                                  self.end_point.y() - center.y()])
                rotated_end = np.dot(rot_matrix, end_vec)
                end_point = QPoint(int(rotated_end[0] + center.x()),
                                 int(rotated_end[1] + center.y()))
            else:
                start_point = self.start_point
                end_point = self.end_point

            # Draw line and endpoint dots
            painter.drawLine(start_point, end_point)
            
            # Draw dots at endpoints
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(start_point, 5, 5)
            painter.drawEllipse(end_point, 5, 5)
            
            painter.end()

        # Scale the pixmap to fit the preview label
        scaled_pixmap = pixmap.scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview.setPixmap(scaled_pixmap)

    def _on_angle_changed(self, value):
        """Handle angle changes from spinbox"""
        self.angle = value
        self.angle_slider.setValue(value)
        self._update_preview()

    def _on_slider_changed(self, value):
        """Handle angle changes from slider"""
        self.angle = value
        self.angle_spin.setValue(value)
        self._update_preview()

    def _on_reset(self):
        """Reset rotation and clear line"""
        self.angle = 0
        self.angle_spin.setValue(0)
        self.angle_slider.setValue(0)
        self.start_point = None
        self.end_point = None
        self._update_preview()

    def _on_confirm(self):
        """Save rotation and proceed"""
        # Store rotation matrix in session if needed
        if self.angle != 0:
            h, w = self.base_frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            self.session.rotation_matrix = M
        
        if self.on_complete:
            self.on_complete()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()
