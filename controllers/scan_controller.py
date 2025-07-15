# controllers/scan_controller.py

import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QProgressBar, QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMetaObject
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

        # Scan buttons side by side
        scan_buttons_layout = QHBoxLayout()
        self.quick_scan_btn = QPushButton("Quick Scan")
        self.quick_scan_btn.clicked.connect(self._on_quick_scan)
        self.quick_scan_btn.setToolTip("Sample 100 frames from the video for faster analysis")
        scan_buttons_layout.addWidget(self.quick_scan_btn)
        
        self.detailed_scan_btn = QPushButton("Detailed Scan")
        self.detailed_scan_btn.clicked.connect(self._on_detailed_scan)
        self.detailed_scan_btn.setToolTip("Analyze every frame in the video (may take longer)")
        scan_buttons_layout.addWidget(self.detailed_scan_btn)
        
        layout.addLayout(scan_buttons_layout)

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
        controls_group.setStyleSheet("QGroupBox { color: black; }")
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

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _handle_error(self, message):
        self.quick_scan_btn.setEnabled(True)
        self.detailed_scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMetaObject.invokeMethod(self, "show_error_message", Qt.QueuedConnection, 
                                 "Scan Error", f"Error during scan: {message}")

    def show_error_message(self, title, message):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, title, message)

    def _on_data_ready(self, time, intensity):
        self.time = time
        self.intensity = intensity
        self.quick_scan_btn.setEnabled(True)
        self.detailed_scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._plot_data()
        
        # Call on_complete callback to notify UI that step is done
        if hasattr(self, 'on_complete') and self.on_complete:
            self.on_complete()
        self._setup_sliders()

    def _plot_data(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.time, self.intensity, 'b-', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        
        # Remove title, legend, and simplify axes
        ax.set_xticks([])  # Remove x-axis ticks and numbers
        ax.set_yticks([])  # Remove y-axis ticks and numbers
        ax.grid(True, alpha=0.3)
        
        # Store references for interactive elements
        self.start_line = ax.axvline(self.time[0], color='red', linewidth=2)
        self.end_line = ax.axvline(self.time[-1], color='red', linewidth=2)
        self.shade = ax.axvspan(self.time[0], self.time[-1], alpha=0.2, color='gray')
        
        self.canvas.draw()

    def _setup_sliders(self):
        if len(self.time) == 0:
            return
            
        # Configure sliders based on data indices
        max_idx = len(self.time) - 1
        
        self.start_slider.setRange(0, max_idx)
        self.start_slider.setValue(0)
        
        self.end_slider.setRange(0, max_idx)
        self.end_slider.setValue(max_idx)
        
        # Update thumbnails
        self._update_thumbnails()

    def _on_start_changed(self, value):
        if len(self.time) == 0:
            return
        self._update_plot_lines()
        self._update_thumbnails()

    def _on_end_changed(self, value):
        if len(self.time) == 0:
            return
        self._update_plot_lines()
        self._update_thumbnails()

    def _update_plot_lines(self):
        if not hasattr(self, 'start_line') or self.start_line is None:
            return
            
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()
        
        if start_idx >= end_idx:
            return
            
        start_time = self.time[start_idx]
        end_time = self.time[end_idx]
        
        self.start_line.set_xdata([start_time])
        self.end_line.set_xdata([end_time])
        
        # Update shaded region
        self.shade.remove()
        ax = self.figure.axes[0]
        self.shade = ax.axvspan(start_time, end_time, alpha=0.2, color='gray')
        
        self.canvas.draw()

    def _update_thumbnails(self):
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()
        
        if hasattr(self.worker, 'indices') and self.worker.indices is not None:
            start_frame = self.worker.indices[start_idx]
            end_frame = self.worker.indices[end_idx]
            
            self._set_thumbnail(self.start_thumb, start_frame)
            self._set_thumbnail(self.end_thumb, end_frame)

    def _set_thumbnail(self, label, frame_idx):
        try:
            frame = self.engine.get_frame(frame_idx)
            if frame is not None:
                # Resize frame to thumbnail size
                h, w = frame.shape[:2]
                thumb_w, thumb_h = self.THUMB_SIZE
                
                # Calculate scaling to maintain aspect ratio
                scale = min(thumb_w / w, thumb_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                resized = cv2.resize(frame, (new_w, new_h))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Create QImage and QPixmap
                bytes_per_line = 3 * new_w
                qt_image = QImage(rgb_frame.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                label.setPixmap(pixmap)
            else:
                self._set_placeholder_thumbnail(label)
        except:
            self._set_placeholder_thumbnail(label)

    def _set_placeholder_thumbnail(self, label):
        # Create a simple placeholder
        pixmap = QPixmap(*self.THUMB_SIZE)
        pixmap.fill(QColor(*self.PLACEHOLDER_COLOR))
        label.setPixmap(pixmap)

    def save_data(self):
        """Save the selected time range to the session"""
        if len(self.time) == 0:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "Please run the scan first.")
            return False
            
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()
        
        if hasattr(self.worker, 'indices') and self.worker.indices is not None:
            self.session.start_frame = self.worker.indices[start_idx]
            self.session.end_frame = self.worker.indices[end_idx]
            return True
        return False

    def _on_quick_scan(self):
        """Perform a quick scan with 100 frames sampled from the video"""
        self._start_scan(max_samples=100)

    def _on_detailed_scan(self):
        """Perform a detailed scan with every frame in the video"""
        self._start_scan(sample_interval=1)

    def _start_scan(self, max_samples=None, sample_interval=None):
        """Start scanning with the specified parameters"""
        # Update worker parameters
        self.worker.max_samples = max_samples
        self.worker.sample_interval = sample_interval
        
        self.quick_scan_btn.setEnabled(False)
        self.detailed_scan_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        if hasattr(self.session, 'start_frame') and hasattr(self.session, 'end_frame'):
            total = self.session.end_frame - self.session.start_frame
            if sample_interval and sample_interval > 1:
                total = total // sample_interval
            elif max_samples and max_samples < total:
                total = max_samples
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
        self.worker.start()

    def closeEvent(self, event):
        """Handle the close event to stop the worker if it's running"""
        # Ensure worker is stopped when navigating away
        if hasattr(self.worker, 'isRunning') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
