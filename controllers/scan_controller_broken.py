# controllers/scan_controller.py

import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QProgressBar, QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from typing import Optional, Callable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class ScanWorker(QThread):
    """Worker thread for scanning video frames."""
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray, np.ndarray)  # time, intensity arrays

    def __init__(self, session, engine, sample_interval=None, max_samples=None):
        super().__init__()
        self.session = session
        self.engine = engine
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.indices = None  # Initialize indices attribute

    def run(self):
        try:
            start, end = self.session.start_frame, self.session.end_frame
            total = end - start
            if total <= 0 or self.engine.cap is None:
                raise RuntimeError("No video loaded")

            if self.sample_interval and self.sample_interval > 1:
                self.indices = np.arange(start, end, self.sample_interval, dtype=int)
            elif self.max_samples and self.max_samples < total:
                self.indices = np.linspace(start, end-1, self.max_samples, dtype=int)
            else:
                self.indices = np.arange(start, end, dtype=int)

            intensities = []
            for i, idx in enumerate(self.indices, start=1):
                frame = self.engine.get_frame(idx)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensities.append(float(gray.mean()))
                self.progress.emit(i)

            base_time = (self.indices - start) / self.session.fps
            time = base_time * getattr(self.session, 'time_multiplier', 1.0)
            self.finished.emit(time, np.array(intensities))
        except Exception as e:
            self.error.emit(str(e))

class ScanController(QWidget):
    """
    Controller for Step 2: Temporal Scan.
    - "Run Scan" for background sampling with determinate progress.
    - Interactive plot with two red vertical lines & grey span.
    - Thumbnails outlined in black beside each slider.
    """

    THUMB_SIZE = (120, 90)
    PLACEHOLDER_COLOR = (220, 220, 220)

    def __init__(
        self,
        parent: QWidget,
        session: VideoSession,
        engine: VideoEngine,
        on_complete=None,
        max_samples: int = 200,  # Default to 200 samples
        sample_interval: int | None = None,
    ):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)
        self.max_samples = max_samples
        self.sample_interval = sample_interval
        
        self.time = np.array([])
        self.intensity = np.array([])
        self.worker = None  # Initialize worker as None

        # Drawing state
        self.drawing = False
        self.start_pos = None
        self.end_pos = None

        self.start_line = None
        self.end_line = None
        self.shade = None

        self._build_ui()
        self._setup_worker()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 2: Temporal Scan")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Click 'Run Scan' to analyze the video frames. "
            "Then adjust the time range using the sliders below the plot."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Run scan button
        self.run_btn = QPushButton("Run Scan")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Plot canvas
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # Controls group
        controls_group = QGroupBox("Time Range Selection")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        # Start frame controls
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start:"))
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.valueChanged.connect(self._on_start_changed)
        start_layout.addWidget(self.start_slider)
        self.start_thumb = QLabel()
        self.start_thumb.setFixedSize(*self.THUMB_SIZE)
        self.start_thumb.setStyleSheet("border: 2px solid black;")
        start_layout.addWidget(self.start_thumb)
        controls_layout.addLayout(start_layout)

        # End frame controls
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End:"))
        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.valueChanged.connect(self._on_end_changed)
        end_layout.addWidget(self.end_slider)
        self.end_thumb = QLabel()
        self.end_thumb.setFixedSize(*self.THUMB_SIZE)
        self.end_thumb.setStyleSheet("border: 2px solid black;")
        end_layout.addWidget(self.end_thumb)
        controls_layout.addLayout(end_layout)

        layout.addWidget(controls_group)

    def _setup_worker(self):
        """Fix worker initialization"""
        self.worker = ScanWorker(
            session=self.session,
            engine=self.engine,
            sample_interval=self.sample_interval,
            max_samples=self.max_samples
        )
        self.worker.progress.connect(self._update_progress)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._on_data_ready)

    def _on_run(self):
        self.run_btn.setEnabled(False)
        self.progress.setMaximum(0)  # Indeterminate progress until we know total
        self.progress.show()
        self.worker.start()

    def _update_progress(self, value):
        if not hasattr(self.worker, 'indices') or self.worker.indices is None:
            self.progress.setMaximum(0)  # Show indeterminate progress
        else:
            self.progress.setMaximum(len(self.worker.indices))
        self.progress.setValue(value)

    def _handle_error(self, error_msg):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Scan Error", error_msg)
        self.run_btn.setEnabled(True)
        self.progress.hide()

    def _on_data_ready(self, time, intensity):
        self.time = time
        self.intensity = intensity
        self.progress.hide()

        self.start_slider.setEnabled(True)
        self.end_slider.setEnabled(True)
        self.start_slider.setRange(0, 1000)  # Using 1000 steps for smooth sliding
        self.end_slider.setRange(0, 1000)
        self.end_slider.setValue(1000)

        self._ax.clear()
        self._ax.plot(self.time, self.intensity)
        self._ax.set_xlabel("Time")
        self._ax.set_ylabel("Light Intensity")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.start_line = self._ax.axvline(0, color='red')
        self.end_line = self._ax.axvline(self.time[-1], color='red')
        self.shade = self._ax.axvspan(0, self.time[-1], color='grey', alpha=0.3)
        self.canvas.draw()

    def _on_slider_move(self):
        if self.start_line is None or self.end_line is None:
            return
        frac_s = self.start_slider.value() / 1000.0
        frac_e = self.end_slider.value() / 1000.0
        t0 = frac_s * self.time[-1]
        t1 = frac_e * self.time[-1]

        self.start_line.set_xdata([t0, t0])
        self.end_line.set_xdata([t1, t1])
        if self.shade:
            self.shade.remove()
        self.shade = self._ax.axvspan(t0, t1, color='grey', alpha=0.3)
        self.canvas.draw()
        self._update_thumbnails()

    def _update_thumbnails(self):
        total = self.session.end_frame - self.session.start_frame
        frac_s = self.start_slider.value() / 1000.0
        frac_e = self.end_slider.value() / 1000.0
        idx_s = self.session.start_frame + int(frac_s * (total-1))
        idx_e = self.session.start_frame + int(frac_e * (total-1))

        def frame_to_pixmap(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qt_image).scaled(*self.THUMB_SIZE, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation)

        frame_s = self.engine.get_frame(idx_s)
        self.start_thumb.setPixmap(frame_to_pixmap(frame_s))

        frame_e = self.engine.get_frame(idx_e)
        self.end_thumb.setPixmap(frame_to_pixmap(frame_e))

    def _on_confirm(self):
        frac_s = self.start_slider.value() / 1000.0
        frac_e = self.end_slider.value() / 1000.0
        total = self.session.end_frame - self.session.start_frame
        new_start = self.session.start_frame + int(frac_s * (total-1))
        new_end = self.session.start_frame + int(frac_e * (total-1))
        self.session.set_frame_range(new_start, new_end)
        self.on_complete()

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
            self._draw_line()

    def _on_mouse_release(self, event):
        """Handle mouse release events on the preview"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pos = event.pos()
            self._draw_line()

    def _draw_line(self):
        """Draw the scan line on the preview"""
        if self.start_pos and self.end_pos:
            # Refresh the base image
            self._update_preview_base()
            
            # Get the current pixmap and draw on it
            pixmap = self.preview.pixmap()
            if pixmap:
                painter = QPainter(pixmap)
                pen = QPen(QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.drawLine(self.start_pos, self.end_pos)
                painter.end()
                self.preview.setPixmap(pixmap)

    def _on_reset(self):
        """Reset the scan line selection"""
        self.start_pos = None
        self.end_pos = None
        self.drawing = False
        self._update_preview_base()

    def _update_preview_base(self):
        """Update the preview with the base frame without any overlays"""
        # Check if video is loaded and session has a start frame
        if (hasattr(self.session, 'start_frame') and 
            self.session.start_frame is not None and 
            hasattr(self.engine, 'cap') and 
            self.engine.cap is not None):
            
            frame = self.engine.get_frame(self.session.start_frame)
            if frame is not None:
                # Apply rotation if it exists
                if hasattr(self.session, 'rotation_matrix') and self.session.rotation_matrix is not None:
                    h, w = frame.shape[:2]
                    cos = np.abs(self.session.rotation_matrix[0, 0])
                    sin = np.abs(self.session.rotation_matrix[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    frame = cv2.warpAffine(frame, self.session.rotation_matrix, (new_w, new_h))
                
                # Convert to RGB for Qt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage and then QPixmap
                h, w = rgb_frame.shape[:2]
                bytes_per_line = 3 * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # Scale pixmap to fit preview while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview.setPixmap(scaled_pixmap)

    def initialize_step(self):
        """Initialize the step when it becomes active"""
        self._update_preview_base()

    def _screen_to_image_pos(self, screen_pos):
        """Convert screen coordinates to image coordinates"""
        if not self.preview.pixmap():
            return screen_pos
        
        pixmap = self.preview.pixmap()
        preview_rect = self.preview.rect()
        pixmap_rect = pixmap.rect()
        
        # Calculate scaling factor
        scale_x = pixmap_rect.width() / preview_rect.width()
        scale_y = pixmap_rect.height() / preview_rect.height()
        
        # Convert coordinates
        image_x = int(screen_pos.x() * scale_x)
        image_y = int(screen_pos.y() * scale_y)
        
        return QPoint(image_x, image_y)

    def save_data(self):
        """Save the scan line to the session"""
        if self.start_pos is None or self.end_pos is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Selection", 
                              "Please draw a line to define the scan region.")
            return False
        
        # Convert screen coordinates to image coordinates
        start = self._screen_to_image_pos(self.start_pos)
        end = self._screen_to_image_pos(self.end_pos)
        
        # Save to session
        self.session.scan_start = start
        self.session.scan_end = end
        return True
