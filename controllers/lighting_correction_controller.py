# controllers/lighting_correction_controller.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

from engine.video_engine import VideoEngine
from models.video_session import VideoSession
from helpers.lighting_correction import sample_background_profile, apply_lighting_correction

class LightingCorrectionController(QWidget):
    """
    Step 6: Lighting Correction
    - Uses a fixed 800×450 canvas.
    - Draw a blank region (blue box), then shows side-by-side preview.
    - Both Apply and Skip simply call on_complete to progress.
    """
    PREVIEW_SIZE = (800, 450)

    def __init__(
        self,
        parent: QWidget,
        session: VideoSession,
        engine: VideoEngine,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)

        # Drawing state
        self.start_point = None
        self.current_point = None
        self.drawing = False
        self.rect = None
        self.preview_image = None

        self._build_ui()
        self._load_frame()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Preview area
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(*self.PREVIEW_SIZE)
        self.preview_label.setMouseTracking(True)
        self.preview_label.mousePressEvent = self._on_mouse_down
        self.preview_label.mouseMoveEvent = self._on_mouse_drag
        self.preview_label.mouseReleaseEvent = self._on_mouse_up
        layout.addWidget(self.preview_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Correction")
        self.apply_btn.clicked.connect(self._on_apply)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._on_skip)
        
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.skip_btn)
        layout.addLayout(btn_layout)

    def _load_frame(self):
        frame = self.engine.get_frame(self.session.start_frame)
        if self.session.rotation_matrix is not None:
            frame = self.engine.apply_rotation(frame, self.session.rotation_matrix)

        h0, w0 = frame.shape[:2]
        pw, ph = self.PREVIEW_SIZE
        scale = min(pw/w0, ph/h0)
        nw, nh = int(w0*scale), int(h0*scale)
        xoff, yoff = (pw-nw)//2, (ph-nh)//2

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage and scale it
        image = QImage(rgb_frame.data, w0, h0, w0 * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Create background pixmap
        self.preview_pixmap = QPixmap(pw, ph)
        self.preview_pixmap.fill(Qt.black)
        
        # Draw scaled image centered
        painter = QPainter(self.preview_pixmap)
        painter.drawPixmap(xoff, yoff, scaled_pixmap)
        painter.end()
        
        self.preview_label.setPixmap(self.preview_pixmap)
        self._scale = scale
        self._xoff = xoff
        self._yoff = yoff

    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_point = QPoint(event.x(), event.y())
        self.current_point = self.start_point

    def _on_mouse_drag(self, event):
        if not self.drawing:
            return
            
        self.current_point = QPoint(event.x(), event.y())
        
        # Create copy of base pixmap and draw rectangle
        preview = self.preview_pixmap.copy()
        painter = QPainter(preview)
        painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue pen
        
        x = min(self.start_point.x(), self.current_point.x())
        y = min(self.start_point.y(), self.current_point.y())
        w = abs(self.current_point.x() - self.start_point.x())
        h = abs(self.current_point.y() - self.start_point.y())
        
        painter.drawRect(x, y, w, h)
        painter.end()
        
        self.preview_label.setPixmap(preview)

    def _on_mouse_up(self, event):
        if not self.drawing:
            return
            
        self.drawing = False
        self.current_point = QPoint(event.x(), event.y())
        
        # Convert to original image coordinates
        x0 = min(self.start_point.x(), self.current_point.x())
        y0 = min(self.start_point.y(), self.current_point.y())
        x1 = max(self.start_point.x(), self.current_point.x())
        y1 = max(self.start_point.y(), self.current_point.y())
        
        ox = int((x0 - self._xoff)/self._scale)
        oy = int((y0 - self._yoff)/self._scale)
        ow = int((x1 - x0)/self._scale)
        oh = int((y1 - y0)/self._scale)
        
        self.rect = (ox, oy, ow, oh)
        self._preview_correction()

    def _preview_correction(self):
        if not self.rect:
            return
            
        profile = sample_background_profile(self.session, self.engine, self.rect)
        mid = (self.session.start_frame + self.session.end_frame)//2
        orig = self.engine.get_frame(mid)
        if self.session.rotation_matrix is not None:
            orig = self.engine.apply_rotation(orig, self.session.rotation_matrix)
        corr = apply_lighting_correction(orig, profile, mid)

        # Create side-by-side preview
        combined = np.concatenate([orig, corr], axis=1)
        h2, w2 = combined.shape[:2]
        pw, ph = self.PREVIEW_SIZE
        scale = min(pw/w2, ph/h2)
        nw, nh = int(w2*scale), int(h2*scale)
        
        # Convert to Qt image
        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Create preview with black background
        preview = QPixmap(pw, ph)
        preview.fill(Qt.black)
        
        # Draw scaled image centered
        xoff, yoff = (pw-nw)//2, (ph-nh)//2
        painter = QPainter(preview)
        painter.drawPixmap(xoff, yoff, scaled_pixmap)
        painter.end()
        
        self.preview_label.setPixmap(preview)

    def _on_apply(self):
        if self.rect is None:
            QMessageBox.critical(self, "Error", "Please draw a blank region first.")
            return
            
        profile = sample_background_profile(self.session, self.engine, self.rect)
        self.session.background_profile = profile
        self.session.background_rect = self.rect
        QMessageBox.information(self, "Lighting Correction", "Background profile stored.")
        self.on_complete()

    def _on_skip(self):
        self.on_complete()
