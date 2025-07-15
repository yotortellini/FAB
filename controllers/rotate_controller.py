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

    def _get_image_coordinates(self, widget_pos):
        """Convert widget coordinates to image coordinates with proper scaling"""
        if self.original_frame is None:
            return None
            
        # Get current pixmap from the preview label
        pixmap = self.preview.pixmap()
        if not pixmap:
            return None
            
        # Get original image dimensions
        img_h, img_w = self.original_frame.shape[:2]
        
        # Get widget dimensions
        widget_rect = self.preview.rect()
        widget_w = widget_rect.width()
        widget_h = widget_rect.height()
        
        # Calculate how the image is displayed (Qt.KeepAspectRatio)
        img_aspect = img_w / img_h
        widget_aspect = widget_w / widget_h
        
        if img_aspect > widget_aspect:
            # Image is wider - limited by widget width
            display_w = widget_w
            display_h = widget_w / img_aspect
        else:
            # Image is taller - limited by widget height
            display_h = widget_h
            display_w = widget_h * img_aspect
            
        # Calculate offset to center the image
        offset_x = (widget_w - display_w) / 2
        offset_y = (widget_h - display_h) / 2
        
        # Check if click is within the displayed image
        if (widget_pos.x() < offset_x or widget_pos.x() > offset_x + display_w or
            widget_pos.y() < offset_y or widget_pos.y() > offset_y + display_h):
            return None
            
        # Convert to image coordinates
        image_x = (widget_pos.x() - offset_x) * img_w / display_w
        image_y = (widget_pos.y() - offset_y) * img_h / display_h
        
        # Clamp to image bounds
        image_x = max(0, min(img_w - 1, image_x))
        image_y = max(0, min(img_h - 1, image_y))
        
        return QPoint(int(image_x), int(image_y))

    def _get_widget_coordinates(self, image_pos):
        """Convert image coordinates to widget coordinates for drawing"""
        if self.original_frame is None:
            return None
            
        # Get original image dimensions
        img_h, img_w = self.original_frame.shape[:2]
        
        # Get widget dimensions
        widget_rect = self.preview.rect()
        widget_w = widget_rect.width()
        widget_h = widget_rect.height()
        
        # Calculate how the image is displayed (Qt.KeepAspectRatio)
        img_aspect = img_w / img_h
        widget_aspect = widget_w / widget_h
        
        if img_aspect > widget_aspect:
            # Image is wider - limited by widget width
            display_w = widget_w
            display_h = widget_w / img_aspect
        else:
            # Image is taller - limited by widget height
            display_h = widget_h
            display_w = widget_h * img_aspect
            
        # Calculate offset to center the image
        offset_x = (widget_w - display_w) / 2
        offset_y = (widget_h - display_h) / 2
        
        # Convert from image coordinates to widget coordinates
        widget_x = image_pos.x() * display_w / img_w + offset_x
        widget_y = image_pos.y() * display_h / img_h + offset_y
        
        return QPoint(int(widget_x), int(widget_y))

    def _on_mouse_press(self, event):
        """Handle mouse press events on the preview"""
        if event.button() == Qt.LeftButton:
            # Clear any previous line
            self.line_points = None
            self.drawing = True
            self.start_pos = event.pos()  # Store widget coordinates for drawing
            self.end_pos = self.start_pos  # Initialize end position
            # Ensure we have the original frame loaded
            if self.original_frame is None:
                self.original_frame = self.engine.get_frame(self.session.start_frame)
            self._update_preview()

    def _on_mouse_move(self, event):
        """Handle mouse move events on the preview"""
        if self.drawing:
            self.end_pos = event.pos()  # Store widget coordinates for drawing
            self._update_preview()

    def _on_mouse_release(self, event):
        """Handle mouse release events on the preview"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pos = event.pos()
            
            # Convert widget coordinates to image coordinates for angle calculation
            start_img = self._get_image_coordinates(self.start_pos)
            end_img = self._get_image_coordinates(self.end_pos)
            
            if start_img and end_img:
                self.line_points = ((start_img.x(), start_img.y()), (end_img.x(), end_img.y()))
                logging.info(f"Widget coords: ({self.start_pos.x()},{self.start_pos.y()}) → ({self.end_pos.x()},{self.end_pos.y()})")
                logging.info(f"Image coords: ({start_img.x()},{start_img.y()}) → ({end_img.x()},{end_img.y()})")
                self._calculate_angle()
                # Automatically apply the rotation after drawing
                self._update_preview()
                
            else:
                # If coordinate conversion failed, clear the line
                self.line_points = None
                self._update_preview()

    def _calculate_angle(self):
        """Calculate rotation angle from the drawn line"""
        if self.line_points:
            (x0, y0), (x1, y1) = self.line_points
            dx = x1 - x0
            dy = y1 - y0
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            # To make the line horizontal, we need to rotate by the negative angle
            correction_angle = -angle_deg
            self.angle_spin.setValue(int(correction_angle))
            self.angle_slider.setValue(int(correction_angle))
            logging.info(f"Line from ({x0},{y0}) to ({x1},{y1}), angle: {angle_deg:.1f}° → correction: {correction_angle:.1f}°")

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
                cv2.line(frame, p0, p1, (0, 255, 0), 5)  # Thicker green line
                cv2.circle(frame, p0, 8, (0, 255, 0), -1)  # Larger green circles
                cv2.circle(frame, p1, 8, (0, 255, 0), -1)
            
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
            
            # If currently drawing, overlay the current line being drawn
            if self.drawing and self.start_pos and self.end_pos:
                painter = QPainter(scaled_pixmap)
                pen = QPen(QColor(255, 0, 0))  # Red line for active drawing
                pen.setWidth(3)  # Visible but not too thick
                painter.setPen(pen)
                
                # Draw line directly using widget coordinates (they map to the scaled pixmap)
                painter.drawLine(self.start_pos, self.end_pos)
                    
                painter.end()
                self.preview.setPixmap(scaled_pixmap)
            else:
                # Set the pixmap without additional drawing
                self.preview.setPixmap(scaled_pixmap)

    def initialize_step(self):
        """Initialize the step when it becomes active"""
        self._update_preview()

    def save_rotation(self):
        """Save the current rotation angle to the session"""
        current_angle = self.angle_spin.value() if hasattr(self, 'angle_spin') else 0
        if current_angle != 0:
            # Use the original frame dimensions, not the rotated frame
            if self.original_frame is not None:
                h, w = self.original_frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
                
                # Compute new bounds for proper translation
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # Adjust translation to keep image centered
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                self.session.rotation_matrix = M
                logging.info(f"Saved rotation matrix for {current_angle}° rotation")
        else:
            # Clear rotation matrix if angle is 0
            self.session.rotation_matrix = None
            logging.info("Cleared rotation matrix (angle = 0)")
        return True  # Always return True since rotation is optional
