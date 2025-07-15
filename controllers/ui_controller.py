# controllers/ui_controller.py

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QStackedWidget, QLabel, QFileDialog,
                            QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from models.video_session import VideoSession
from engine.video_engine import VideoEngine
from .preview_controller import PreviewController
from .scan_controller import ScanController
from .rotate_controller import RotateController
from .roi_controller import ROIController
from .analysis_controller import AnalysisController

from pathlib import Path

class UIController(QMainWindow):
    """
    Main window controller that manages the workflow:
    1. Preview video
    2. Temporal scan
    3. Rotate/crop
    4. Define ROIs
    5. Analysis
    """

    def __init__(self, session: VideoSession, engine: VideoEngine):
        super().__init__()
        self.session = session
        self.engine = engine
        self.current_step = 1  # Start at step 1 (not 0)
        self.total_steps = 5
        
        self._build_ui()
        self._setup_navigation()
        
        # Set window title
        self.setWindowTitle("Flow Analysis Lab (FAB)")
        
        # Set initial window size to 2/3 of current size
        self.resize(800, 600)

    def _build_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header with logo
        header_layout = QHBoxLayout()
        self.logo_label = QLabel()
        header_layout.addStretch()
        header_layout.addWidget(self.logo_label)
        main_layout.addLayout(header_layout)

        # Stacked widget for different steps
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._on_back)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._on_next)
        
        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)
        main_layout.addLayout(nav_layout)

        # Initialize controllers
        self.preview_ctrl = PreviewController(
            self.stack, self.session, self.engine,
            on_complete=lambda: self.show_step(2)
        )
        self.scan_ctrl = ScanController(
            self.stack, self.session, self.engine,
            on_complete=lambda: self.show_step(3)
        )
        self.rotate_ctrl = RotateController(
            self.stack, self.session, self.engine,
            on_complete=lambda: self.show_step(4)
        )
        self.roi_ctrl = ROIController(
            self.stack, self.session, self.engine,
            on_complete=lambda: self.show_step(5)
        )
        self.analysis_ctrl = AnalysisController(
            self.stack, self.session, self.engine,
            on_complete=lambda: self.show_step(1)
        )

        # Add controllers to stack
        self.stack.addWidget(self.preview_ctrl)
        self.stack.addWidget(self.scan_ctrl)
        self.stack.addWidget(self.rotate_ctrl)
        self.stack.addWidget(self.roi_ctrl)
        self.stack.addWidget(self.analysis_ctrl)

        # Show first step
        self.show_step(1)

    def _setup_navigation(self):
        """Configure navigation button states"""
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def _load_logo(self):
        """Load and display the logo"""
        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled_pixmap = pixmap.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)

    def show_step(self, step: int):
        """Switch to the specified step in the workflow"""
        # Update stack (subtract 1 because stack is 0-based)
        self.stack.setCurrentIndex(step - 1)
        self.current_step = step
        
        # Update navigation buttons
        self._update_navigation()
        
    def _update_navigation(self):
        """Update navigation button states based on current step"""
        # Enable back button if we're not on the first step
        self.back_button.setEnabled(self.current_step > 1)
        
        # Hide next button on the last step (analysis)
        if self.current_step == self.total_steps:
            self.next_button.hide()
        else:
            self.next_button.show()
            self.next_button.setEnabled(self.is_step_complete(self.current_step))

    def is_step_complete(self, step: int) -> bool:
        """Check if the current step is complete and we can proceed"""
        if step == 1:  # Video Preview
            return self.engine.cap is not None
        elif step == 2:  # Temporal Scan
            return hasattr(self.scan_ctrl, 'time') and len(self.scan_ctrl.time) > 0
        elif step == 3:  # Rotate
            return True  # Rotation is optional
        elif step == 4:  # ROI Definition
            return len(self.session.rois) > 0
        return False

    def _on_back(self):
        """Handle back button click"""
        if self.current_step > 1:
            self.show_step(self.current_step - 1)

    def _on_next(self):
        """Handle next button click"""
        if self.current_step < self.total_steps:
            self.show_step(self.current_step + 1)

    def browse_video(self):
        """Open file dialog to select a video file"""
        # Use getOpenFileName with an explicit parent and native dialog
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Video File",
            directory=str(Path.home()),  # Start in user's home directory
            filter="Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
            options=QFileDialog.Options()  # Use native dialog
        )
        
        if file_path:
            try:
                self.engine.load_video(file_path)
                self.preview_ctrl._setup_preview()
                self.next_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
        else:
            # If no file was selected, show message but don't quit
            QMessageBox.information(
                self,
                "No File Selected",
                "Please use the 'Load Video' button to select a video file when ready."
            )