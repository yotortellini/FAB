# controllers/scan_controller.py

import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QProgressBar, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
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

        # Plot area
        fig = Figure(figsize=(6, 3))
        self._ax = fig.add_subplot(111)
        self._ax.set_xlabel("Time")
        self._ax.set_ylabel("Light Intensity")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(self.canvas)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)

        # Run Scan button
        self.run_btn = QPushButton("Run Scan")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)

        # Start slider + thumbnail
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Measurement Start"))
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setEnabled(False)
        self.start_slider.valueChanged.connect(self._on_slider_move)
        start_layout.addWidget(self.start_slider)
        self.start_thumb = QLabel()
        self.start_thumb.setFixedSize(*self.THUMB_SIZE)
        self.start_thumb.setStyleSheet("border: 1px solid black;")
        start_layout.addWidget(self.start_thumb)
        layout.addLayout(start_layout)

        # End slider + thumbnail
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("Measurement End"))
        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setEnabled(False)
        self.end_slider.valueChanged.connect(self._on_slider_move)
        end_layout.addWidget(self.end_slider)
        self.end_thumb = QLabel()
        self.end_thumb.setFixedSize(*self.THUMB_SIZE)
        self.end_thumb.setStyleSheet("border: 1px solid black;")
        end_layout.addWidget(self.end_thumb)
        layout.addLayout(end_layout)

        # Set placeholder images
        placeholder = QImage(*self.THUMB_SIZE, QImage.Format_RGB888)
        placeholder.fill(Qt.lightGray)
        placeholder_pixmap = QPixmap.fromImage(placeholder)
        self.start_thumb.setPixmap(placeholder_pixmap)
        self.end_thumb.setPixmap(placeholder_pixmap)

        # Confirm Selection button
        self.confirm_btn = QPushButton("Confirm Selection")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(self.confirm_btn)

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

        self.confirm_btn.setEnabled(True)

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
