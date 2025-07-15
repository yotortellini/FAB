# controllers/preview_controller.py

import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QFrame, QFileDialog, QMessageBox, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from pathlib import Path

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class PreviewController(QWidget):
    """
    Controller for Step 1: Video Preview
    Displays the first frame of the video with basic info.
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
        
        self._build_ui()
        self._setup_preview()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 1: Video Preview")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Load button area with some styling
        load_area = QHBoxLayout()
        self.load_btn = QPushButton("Load Video File")
        self.load_btn.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                padding: 10px 20px;
                min-width: 200px;
            }
        """)
        self.load_btn.clicked.connect(self._on_load)
        load_area.addStretch()
        load_area.addWidget(self.load_btn)
        load_area.addStretch()
        layout.addLayout(load_area)

        # Preview frame with placeholder text
        self.preview = QLabel("No video loaded. Click 'Load Video File' to begin.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                color: #666;
                border: 2px dashed #ccc;
                border-radius: 5px;
                padding: 20px;
            }
        """)
        self.preview.setMinimumSize(640, 480)
        layout.addWidget(self.preview)

        # Video info
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.info_label)

    def _setup_preview(self):
        """Set up the preview display and video information"""
        if not self.engine.cap:
            self.info_label.setText("No video loaded")
            return False

        # Get video properties
        fps = self.engine.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.engine.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.engine.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.engine.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update info text
        duration = frame_count / fps
        info_text = (
            f"Video Information:\n"
            f"Resolution: {width}x{height}\n"
            f"Frame Rate: {fps:.1f} fps\n"
            f"Duration: {duration:.1f} seconds\n"
            f"Total Frames: {frame_count}"
        )
        self.info_label.setText(info_text)

        # Display first frame
        frame = self.engine.get_frame(0)
        if frame is not None:
            # Convert frame to QPixmap and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale pixmap to fit the preview frame while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.preview.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.preview.setPixmap(scaled_pixmap)

            # Store video properties in session
            self.session.fps = fps
            self.session.frame_count = frame_count
            self.session.set_frame_range(0, frame_count - 1)
            
            return True
            
        return False

    def _on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Video File",
            directory=str(Path.home()),
            filter="Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
            options=QFileDialog.Options()
        )
        
        if file_path:
            try:
                self.engine.load_video(file_path)
                self._setup_preview()
                
                # Enable the Next button in the main window
                main_window = self.window()
                if hasattr(main_window, '_update_navigation'):
                    main_window._update_navigation()
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")

    def resizeEvent(self, event):
        """Handle window resize by scaling the preview image."""
        super().resizeEvent(event)
        if hasattr(self, 'preview') and self.preview.pixmap():
            # Re-scale the current pixmap to fit the new size
            frame = self.engine.get_frame(0)
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.preview.size(), 
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.preview.setPixmap(scaled_pixmap)
