# controllers/rotate_controller.py

import cv2
import numpy as np
import math
import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QSlider, QGroupBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from typing import Optional, Callable

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class DrawableLabel(QLabel):
    """Custom QLabel that can draw overlay lines"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.line_start = None
        self.line_end = None
        self.drawing = False
        self.line_fixed = False
        
    def set_line(self, start, end, drawing=False, fixed=False):
        """Update line state and trigger repaint"""
        self.line_start = start
        self.line_end = end
        self.drawing = drawing
        self.line_fixed = fixed
        self.update()  # Trigger paintEvent
        logging.info(f"DrawableLabel.set_line: start={self.line_start}, end={self.line_end}, drawing={self.drawing}, fixed={self.line_fixed}")
        
    def clear_line(self):
        """Clear the line and repaint"""
        self.line_start = None
        self.line_end = None
        self.drawing = False
        self.line_fixed = False
        self.update()
        
    def paintEvent(self, event):
        """Override to draw line overlay"""
        # First draw the base pixmap
        super().paintEvent(event)
        
        # Then draw line overlay if we have valid points
        if self.drawing and self.line_start and self.line_end:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0))  # Bright red
            pen.setWidth(3)
            painter.setPen(pen)
            
            # Draw line directly using widget coordinates
            painter.drawLine(self.line_start, self.line_end)
            
            # Draw endpoints as small red circles
            brush = QBrush(QColor(255, 0, 0))
            painter.setBrush(brush)
            dot_radius = 5
            painter.drawEllipse(self.line_start.x() - dot_radius,
                                self.line_start.y() - dot_radius,
                                dot_radius * 2, dot_radius * 2)
            painter.drawEllipse(self.line_end.x() - dot_radius,
                                self.line_end.y() - dot_radius,
                                dot_radius * 2, dot_radius * 2)
            
            painter.end()

class RotateController(QWidget):
    """
    Controller for Step 3: Rotate Video
    Simple approach: Draw a red line on the preview, calculate angle, auto-rotate
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
        
        # Simple state variables
        self.current_angle = 0
        self.base_frame = None  # Original unrotated frame
        self.rotated_frame = None  # Current rotated frame
        
        # Line drawing state
        self.drawing = False
        self.line_start_widget = None  # QPoint in widget coordinates (for drawing)
        self.line_end_widget = None    # QPoint in widget coordinates (for drawing)
        self.line_start_image = None   # QPoint in image coordinates (persistent)
        self.line_end_image = None     # QPoint in image coordinates (persistent)
        self.line_fixed = False # True when line is drawn and angle calculated
        
        # Prevent infinite loops during UI updates
        self._updating_controls = False
        
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 3: Rotate Video")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Draw a RED line across a feature that should be horizontal. "
            "The image will automatically rotate to make your line horizontal."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Preview frame with mouse handling
        self.preview = DrawableLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        self.preview.setText("Load a video to start rotation...")
        
        # Enable mouse events
        self.preview.setMouseTracking(True)
        self.preview.mousePressEvent = self._on_mouse_press
        self.preview.mouseMoveEvent = self._on_mouse_move
        self.preview.mouseReleaseEvent = self._on_mouse_release
        
        layout.addWidget(self.preview)

        # Controls
        controls_group = QGroupBox("Rotation Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        # Angle display and manual adjustment
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Rotation Angle:"))
        
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(-180, 180)
        self.angle_spin.setSuffix("°")
        self.angle_spin.valueChanged.connect(self._on_manual_angle_change)
        angle_layout.addWidget(self.angle_spin)

        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._on_slider_change)
        angle_layout.addWidget(self.angle_slider)
        
        controls_layout.addLayout(angle_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset Rotation")
        self.reset_btn.clicked.connect(self._reset_rotation)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        button_layout.addWidget(self.reset_btn)
        
        self.clear_line_btn = QPushButton("Clear Line")
        self.clear_line_btn.clicked.connect(self._clear_line)
        self.clear_line_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #e68900; }
        """)
        button_layout.addWidget(self.clear_line_btn)
        
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)
        
        layout.addWidget(controls_group)

    def initialize_step(self):
        """Load the base frame and display it"""
        if self.session.start_frame is not None and self.engine.cap is not None:
            self.base_frame = self.engine.get_frame(self.session.start_frame)
            if self.base_frame is not None:
                logging.info("Loaded base frame for rotation")
                self._update_display()
            else:
                logging.error("Failed to load base frame")
        else:
            logging.warning("No video loaded for rotation step")

    def _widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if self.base_frame is None or not self.preview.pixmap():
            return None
            
        # Get current pixmap dimensions
        pixmap = self.preview.pixmap()
        widget_rect = self.preview.rect()
        
        # Calculate how image is scaled and positioned
        pixmap_rect = pixmap.rect()
        
        # Calculate scale factor (Qt.KeepAspectRatio)
        widget_aspect = widget_rect.width() / widget_rect.height()
        pixmap_aspect = pixmap_rect.width() / pixmap_rect.height()
        
        if pixmap_aspect > widget_aspect:
            # Image limited by widget width
            scale = widget_rect.width() / pixmap_rect.width()
            scaled_width = widget_rect.width()
            scaled_height = pixmap_rect.height() * scale
        else:
            # Image limited by widget height
            scale = widget_rect.height() / pixmap_rect.height()
            scaled_width = pixmap_rect.width() * scale
            scaled_height = widget_rect.height()
        
        # Calculate offset to center image
        offset_x = (widget_rect.width() - scaled_width) / 2
        offset_y = (widget_rect.height() - scaled_height) / 2
        
        # Check if click is within image bounds
        if (widget_pos.x() < offset_x or widget_pos.x() > offset_x + scaled_width or
            widget_pos.y() < offset_y or widget_pos.y() > offset_y + scaled_height):
            return None
            
        # Convert to image coordinates
        image_x = (widget_pos.x() - offset_x) / scale
        image_y = (widget_pos.y() - offset_y) / scale
        
        return QPoint(int(image_x), int(image_y))

    def _image_to_widget_coords(self, image_pos):
        """Convert image coordinates to widget coordinates"""
        if self.base_frame is None or not self.preview.pixmap():
            return None
            
        # Get current pixmap dimensions
        pixmap = self.preview.pixmap()
        widget_rect = self.preview.rect()
        
        # Calculate how image is scaled and positioned
        pixmap_rect = pixmap.rect()
        
        # Calculate scale factor (Qt.KeepAspectRatio)
        widget_aspect = widget_rect.width() / widget_rect.height()
        pixmap_aspect = pixmap_rect.width() / pixmap_rect.height()
        
        if pixmap_aspect > widget_aspect:
            # Image limited by widget width
            scale = widget_rect.width() / pixmap_rect.width()
            scaled_width = widget_rect.width()
            scaled_height = pixmap_rect.height() * scale
        else:
            # Image limited by widget height
            scale = widget_rect.height() / pixmap_rect.height()
            scaled_width = pixmap_rect.width() * scale
            scaled_height = widget_rect.height()
        
        # Calculate offset to center image
        offset_x = (widget_rect.width() - scaled_width) / 2
        offset_y = (widget_rect.height() - scaled_height) / 2
        
        # Convert to widget coordinates
        widget_x = image_pos.x() * scale + offset_x
        widget_y = image_pos.y() * scale + offset_y
        
        return QPoint(int(widget_x), int(widget_y))

    def _on_mouse_press(self, event):
        """Start drawing a line"""
        if event.button() == Qt.LeftButton and self.base_frame is not None:
            self.drawing = True
            self.line_fixed = False
            
            # Store both widget and image coordinates (now on the base frame)
            self.line_start_widget = event.pos()
            self.line_end_widget = event.pos()
            self.line_start_image = self._widget_to_image_coords(event.pos())
            self.line_end_image = self.line_start_image
            
            # Update preview line
            self.preview.set_line(self.line_start_widget, self.line_end_widget, drawing=True, fixed=False)
            logging.info(f"Started drawing line at widget {self.line_start_widget.x()}, {self.line_start_widget.y()}")

    def _on_mouse_move(self, event):
        """Update line end point while drawing"""
        if self.drawing:
            self.line_end_widget = event.pos()
            self.line_end_image = self._widget_to_image_coords(event.pos())
            self.preview.set_line(self.line_start_widget, self.line_end_widget, drawing=True, fixed=False)

    def _on_mouse_release(self, event):
        """Finish drawing line and calculate rotation"""
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.line_end_widget = event.pos()
            self.line_end_image = self._widget_to_image_coords(event.pos())
            self.line_fixed = True

            # Cache coordinates for logging after clearing the line
            _start_widget = self.line_start_widget
            _end_widget = self.line_end_widget

            # Calculate the angle and apply rotation
            self._calculate_and_apply_rotation()

            # Clear the drawn line now that rotation is applied
            self._clear_line()

            logging.info(f"Finished drawing line from widget {_start_widget.x()}, {_start_widget.y()} to {_end_widget.x()}, {_end_widget.y()}")

    def _calculate_and_apply_rotation(self):
        """Calculate rotation angle from the drawn line and apply it"""
        if not self.line_start_image or not self.line_end_image:
            return

        # Calculate angle using IMAGE coordinates (more accurate)
        dx = self.line_end_image.x() - self.line_start_image.x()
        dy = self.line_end_image.y() - self.line_start_image.y()

        if dx == 0 and dy == 0:
            return  # No line drawn

        # Calculate angle in degrees
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # To make the line horizontal, we need to rotate by the negative of the line angle
        correction_angle = angle_deg

        # Add this correction to the current angle (accumulative rotation)
        self.current_angle += correction_angle

        # Normalize to -180 to 180
        while self.current_angle > 180:
            self.current_angle -= 360
        while self.current_angle < -180:
            self.current_angle += 360

        # Update UI controls with signal blocking to prevent loops
        self._updating_controls = True
        self.angle_spin.setValue(int(self.current_angle))
        self.angle_slider.setValue(int(self.current_angle))
        self._updating_controls = False

        # Apply rotation and update display
        self._apply_rotation()

        logging.info(f"Calculated rotation: line angle {angle_deg:.1f}, correction {correction_angle:.1f}, total angle {self.current_angle:.1f}")

    def _on_manual_angle_change(self, value):
        """Handle manual angle changes from spinbox"""
        if self._updating_controls:
            return
        self.current_angle = value
        self._updating_controls = True
        self.angle_slider.setValue(value)
        self._updating_controls = False
        self._apply_rotation()

    def _on_slider_change(self, value):
        """Handle angle changes from slider"""
        if self._updating_controls:
            return
        self.current_angle = value
        self._updating_controls = True
        self.angle_spin.setValue(value)
        self._updating_controls = False
        self._apply_rotation()

    def _apply_rotation(self):
        """Apply the current rotation angle to the base frame"""
        if self.base_frame is None:
            return
            
        if abs(self.current_angle) < 0.1:  # Essentially no rotation
            self.rotated_frame = self.base_frame.copy()
        else:
            # Get image center and create rotation matrix
            h, w = self.base_frame.shape[:2]
            center = (w // 2, h // 2)
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, self.current_angle, 1.0)
            
            # Calculate new bounding box to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust translation to center the image
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation
            self.rotated_frame = cv2.warpAffine(self.base_frame, M, (new_w, new_h))
            
        self._update_display()

    def _update_display(self):
        """Update the preview with current frame and update line position"""
        if self.base_frame is None:
            return
            
        # Use rotated frame if available, otherwise base frame
        display_frame = self.rotated_frame if self.rotated_frame is not None else self.base_frame
        
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage and QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit preview
        preview_size = self.preview.size()
        scaled_pixmap = pixmap.scaled(preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap first
        self.preview.setPixmap(scaled_pixmap)
        
        # Update line position if we have a fixed line
        if self.line_fixed and self.line_start_image and self.line_end_image:
            # Transform the line coordinates through the rotation matrix
            if abs(self.current_angle) > 0.1:
                # Apply the same rotation transformation to the line coordinates
                h_orig, w_orig = self.base_frame.shape[:2]
                center = (w_orig // 2, h_orig // 2)
                
                # Create rotation matrix (same as used for image)
                M = cv2.getRotationMatrix2D(center, self.current_angle, 1.0)
                
                # Calculate new bounds (same as used for image)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h_orig * sin) + (w_orig * cos))
                new_h = int((h_orig * cos) + (w_orig * sin))
                
                # Adjust translation (same as used for image)
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                # Transform line start and end points
                start_homog = np.array([self.line_start_image.x(), self.line_start_image.y(), 1])
                end_homog = np.array([self.line_end_image.x(), self.line_end_image.y(), 1])
                
                transformed_start = M.dot(start_homog)
                transformed_end = M.dot(end_homog)
                
                # Convert back to image coordinates in the rotated space
                rotated_start_image = QPoint(int(transformed_start[0]), int(transformed_start[1]))
                rotated_end_image = QPoint(int(transformed_end[0]), int(transformed_end[1]))
            else:
                # No rotation, use original coordinates
                rotated_start_image = self.line_start_image
                rotated_end_image = self.line_end_image
            
            # Now convert the rotated image coordinates to widget coordinates
            new_start_widget = self._image_to_widget_coords_for_display(rotated_start_image, scaled_pixmap)
            new_end_widget = self._image_to_widget_coords_for_display(rotated_end_image, scaled_pixmap)
            
            logging.info(f"_update_display: transformed widget coords start={new_start_widget}, end={new_end_widget}")
            if new_start_widget and new_end_widget:
                # Update the line position to match the rotated image
                self.preview.set_line(new_start_widget, new_end_widget, drawing=False, fixed=True)
            else:
                # Clear line if coordinates are invalid
                self.preview.clear_line()

    def _image_to_widget_coords_for_display(self, image_pos, scaled_pixmap):
        """Convert image coordinates to widget coordinates for the current display"""
        if not scaled_pixmap:
            return None
            
        # Get widget dimensions
        widget_rect = self.preview.rect()
        
        # Get scaled pixmap dimensions
        pixmap_w = scaled_pixmap.width()
        pixmap_h = scaled_pixmap.height()
        
        # Get the current displayed image dimensions (rotated frame)
        display_frame = self.rotated_frame if self.rotated_frame is not None else self.base_frame
        img_h, img_w = display_frame.shape[:2]
        
        # Calculate scale factor from image to scaled pixmap
        scale_x = pixmap_w / img_w
        scale_y = pixmap_h / img_h
        
        # Convert image coordinates to pixmap coordinates
        pixmap_x = image_pos.x() * scale_x
        pixmap_y = image_pos.y() * scale_y
        
        # Calculate offset to center scaled image in widget
        offset_x = (widget_rect.width() - pixmap_w) / 2
        offset_y = (widget_rect.height() - pixmap_h) / 2
        
        # Convert to widget coordinates
        widget_x = pixmap_x + offset_x
        widget_y = pixmap_y + offset_y
        
        return QPoint(int(widget_x), int(widget_y))

    def _reset_rotation(self):
        """Reset rotation to 0 degrees"""
        self.current_angle = 0
        self.angle_spin.setValue(0)
        self.angle_slider.setValue(0)
        self._apply_rotation()
        logging.info("Reset rotation to 0°")

    def _clear_line(self):
        """Clear the drawn line"""
        self.line_start_widget = None
        self.line_end_widget = None
        self.line_start_image = None
        self.line_end_image = None
        self.line_fixed = False
        self.drawing = False
        self.preview.clear_line()
        logging.info("Cleared rotation line")

    def save_rotation(self):
        """Save the rotation matrix to the session"""
        if abs(self.current_angle) > 0.1:  # Only save if there's meaningful rotation
            h, w = self.base_frame.shape[:2]
            center = (w // 2, h // 2)
            
            # Create the same rotation matrix used for display
            M = cv2.getRotationMatrix2D(center, self.current_angle, 1.0)
            
            # Calculate new bounds
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            self.session.rotation_matrix = M
            logging.info(f"Saved rotation matrix for {self.current_angle:.1f}° rotation")
        else:
            self.session.rotation_matrix = None
            logging.info("No rotation to save (angle near 0°)")
        
        return True
