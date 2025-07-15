# controllers/roi_controller.py

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QLineEdit, QListWidget, QMessageBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

from models.video_session import VideoSession
from models.roi import ROI
from engine.video_engine import VideoEngine

class ROIController(QWidget):
    """
    Controller for Step 4: ROI Definition
    Allows drawing and naming rectangular ROIs on the video frame.
    """

    def __init__(
        self,
        parent: QWidget,
        session: VideoSession,
        engine: VideoEngine,
        on_complete=None
    ):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)

        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.rois = {}  # name -> ROI
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 4: Define ROIs")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setMouseTracking(True)
        self.frame_label.mousePressEvent = self._on_mouse_press
        self.frame_label.mouseMoveEvent = self._on_mouse_move
        self.frame_label.mouseReleaseEvent = self._on_mouse_release
        layout.addWidget(self.frame_label)

        # ROI controls
        controls = QHBoxLayout()
        
        # ROI list
        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._on_roi_selected)
        controls.addWidget(self.roi_list)

        # ROI input area
        input_layout = QVBoxLayout()
        
        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("ROI Name:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        input_layout.addLayout(name_layout)

        # Volume input
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume (µL):"))
        self.volume_input = QLineEdit()
        volume_layout.addWidget(self.volume_input)
        input_layout.addLayout(volume_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add ROI")
        self.add_btn.clicked.connect(self._on_add)
        self.delete_btn = QPushButton("Delete ROI")
        self.delete_btn.clicked.connect(self._on_delete)
        self.confirm_btn = QPushButton("Confirm ROIs")
        self.confirm_btn.clicked.connect(self._on_confirm)
        
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.confirm_btn)
        
        input_layout.addLayout(button_layout)
        controls.addLayout(input_layout)
        layout.addLayout(controls)

        # Initial setup
        self._update_frame()
        self.delete_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)

    def _get_image_position(self, pos):
        """Convert screen coordinates to image coordinates"""
        if not self.frame_label.pixmap():
            return None

        # Get the scaled pixmap that's actually displayed
        pixmap = self.frame_label.pixmap()
        label_rect = self.frame_label.rect()
        scaled_rect = pixmap.rect()
        scaled_rect.moveCenter(label_rect.center())

        # Check if click is within the image area
        if not scaled_rect.contains(pos):
            return None

        # Calculate the scaling factors
        original_size = self.engine.get_frame(self.session.start_frame).shape[:2]
        scale_x = original_size[1] / scaled_rect.width()
        scale_y = original_size[0] / scaled_rect.height()

        # Convert screen coordinates to original image coordinates
        image_x = (pos.x() - scaled_rect.left()) * scale_x
        image_y = (pos.y() - scaled_rect.top()) * scale_y

        return QPoint(int(image_x), int(image_y))

    def _on_mouse_press(self, event):
        pos = self._get_image_position(event.pos())
        if pos and event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = pos
            self.current_point = pos
            self._update_frame()

    def _on_mouse_move(self, event):
        if self.drawing:
            pos = self._get_image_position(event.pos())
            if pos:
                self.current_point = pos
                self._update_frame()

    def _on_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            pos = self._get_image_position(event.pos())
            if pos:
                self.current_point = pos
                self._update_frame()

    def _on_add(self):
        """Add a new ROI with the current drawing"""
        if not (self.start_point and self.current_point):
            QMessageBox.warning(self, "Error", "Please draw an ROI first")
            return

        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter an ROI name")
            return

        if name in self.rois:
            QMessageBox.warning(self, "Error", "An ROI with this name already exists")
            return

        try:
            volume = float(self.volume_input.text() or 0)
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid volume number")
            return

        # Calculate ROI bounds in image coordinates
        x = min(self.start_point.x(), self.current_point.x())
        y = min(self.start_point.y(), self.current_point.y())
        w = abs(self.current_point.x() - self.start_point.x())
        h = abs(self.current_point.y() - self.start_point.y())

        try:
            # Create ROI with default values for analysis parameters
            roi = ROI(
                name=name,
                rect=(int(x), int(y), int(w), int(h)),
                channel='auto',
                total_volume=volume,
                start_frac=0.0,
                end_frac=1.0,
                interval=1,
                smoothing_window=5
            )
            self.rois[name] = roi
            
            # Update UI
            self.roi_list.addItem(name)
            self.name_input.clear()
            self.volume_input.clear()
            self.start_point = None
            self.current_point = None
            
            # Enable buttons
            self.delete_btn.setEnabled(True)
            self.confirm_btn.setEnabled(len(self.rois) > 0)
            
            self._update_frame()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create ROI: {str(e)}")

    def _on_delete(self):
        current = self.roi_list.currentItem()
        if current:
            name = current.text()
            del self.rois[name]
            self.roi_list.takeItem(self.roi_list.row(current))
            self.confirm_btn.setEnabled(len(self.rois) > 0)
            self._update_frame()

    def _on_roi_selected(self, row):
        self.delete_btn.setEnabled(row >= 0)

    def _on_confirm(self):
        self.session.rois = self.rois
        self.on_complete()

    def _update_frame(self):
        """Update the frame display with current video frame and ROIs"""
        if not self.engine.cap:
            return

        # Get the frame
        frame = self.engine.get_frame(self.session.start_frame)
        if frame is None:
            return

        # Convert frame to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Create a painter for drawing
        painter = QPainter(pixmap)
        
        # Draw existing ROIs
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        for name, roi in self.rois.items():
            x, y, w, h = roi.rect
            painter.drawRect(x, y, w, h)
            # Draw name above the ROI
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(x, y - 5, name)

        # Draw current ROI if drawing
        if self.drawing and self.start_point and self.current_point:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            x = min(self.start_point.x(), self.current_point.x())
            y = min(self.start_point.y(), self.current_point.y())
            w = abs(self.current_point.x() - self.start_point.x())
            h = abs(self.current_point.y() - self.start_point.y())
            painter.drawRect(int(x), int(y), int(w), int(h))

        painter.end()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.frame_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled_pixmap)
        self.frame_label.setAlignment(Qt.AlignCenter)
