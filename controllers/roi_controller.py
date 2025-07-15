# controllers/roi_controller.py

import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QLineEdit, QListWidget, QMessageBox, QGroupBox,
                            QSlider, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

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

        # Initialize UI controller reference
        self.ui_controller = None

        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.pending_roi = None  # Store the last drawn rectangle until it's added or replaced
        self.rois = {}  # name -> ROI
        self._build_ui()

        # Initialize Next button
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(True)

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 4: Define ROIs")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Click and drag to draw rectangular regions of interest (ROIs) on the video frame. "
            "Each ROI represents an area where flow analysis will be performed."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Main content area with consistent layout
        content_layout = QHBoxLayout()
        
        # Left side: Frame display (similar to other steps)
        frame_container = QVBoxLayout()
        
        # Frame display with proper sizing and styling
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
        self.frame_label.setText("Loading video frame...")
        self.frame_label.setMouseTracking(True)
        self.frame_label.mousePressEvent = self._on_mouse_press
        self.frame_label.mouseMoveEvent = self._on_mouse_move
        self.frame_label.mouseReleaseEvent = self._on_mouse_release
        frame_container.addWidget(self.frame_label)
        
        content_layout.addLayout(frame_container, 3)  # Give more space to frame
        
        # Right side: Controls (organized like other steps)
        controls_container = QVBoxLayout()
        
        # ROI Creation Group
        creation_group = QGroupBox("Create New ROI")
        creation_group.setStyleSheet("QGroupBox::title { color: black; font-weight: bold; }")
        creation_layout = QVBoxLayout()
        creation_group.setLayout(creation_layout)
        
        # Instructions for ROI creation
        roi_instructions = QLabel("1. Click and drag on the frame to draw a rectangle\n2. Enter a name for the ROI\n3. Click 'Add ROI' to save")
        roi_instructions.setStyleSheet("color: #666; font-style: italic;")
        creation_layout.addWidget(roi_instructions)
        
        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("ROI Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Channel_1, Outlet, etc.")
        name_layout.addWidget(self.name_input)
        creation_layout.addLayout(name_layout)

        # Volume input
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume (µL):"))
        self.volume_input = QLineEdit()
        self.volume_input.setPlaceholderText("e.g., 50, 100, 250")
        volume_layout.addWidget(self.volume_input)
        creation_layout.addLayout(volume_layout)

        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Color Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["auto", "R", "G", "B"])
        self.channel_combo.setCurrentText("auto")
        self.channel_combo.setToolTip("Select which color channel to measure intensity from")
        channel_layout.addWidget(self.channel_combo)
        creation_layout.addLayout(channel_layout)

        # Start Fraction input
        start_frac_layout = QVBoxLayout()
        start_frac_label = QLabel("Start Fraction (0.0 - 1.0):")
        start_frac_layout.addWidget(start_frac_label)
        
        start_frac_controls = QHBoxLayout()
        self.start_frac_slider = QSlider(Qt.Horizontal)
        self.start_frac_slider.setMinimum(0)
        self.start_frac_slider.setMaximum(100)
        self.start_frac_slider.setValue(0)
        self.start_frac_slider.valueChanged.connect(self._on_start_frac_slider_changed)
        
        self.start_frac_spin = QDoubleSpinBox()
        self.start_frac_spin.setMinimum(0.0)
        self.start_frac_spin.setMaximum(1.0)
        self.start_frac_spin.setSingleStep(0.01)
        self.start_frac_spin.setDecimals(2)
        self.start_frac_spin.setValue(0.0)
        self.start_frac_spin.valueChanged.connect(self._on_start_frac_spin_changed)
        
        start_frac_controls.addWidget(self.start_frac_slider, 3)
        start_frac_controls.addWidget(self.start_frac_spin, 1)
        start_frac_layout.addLayout(start_frac_controls)
        creation_layout.addLayout(start_frac_layout)

        # End Fraction input
        end_frac_layout = QVBoxLayout()
        end_frac_label = QLabel("End Fraction (0.0 - 1.0):")
        end_frac_layout.addWidget(end_frac_label)
        
        end_frac_controls = QHBoxLayout()
        self.end_frac_slider = QSlider(Qt.Horizontal)
        self.end_frac_slider.setMinimum(0)
        self.end_frac_slider.setMaximum(100)
        self.end_frac_slider.setValue(100)
        self.end_frac_slider.valueChanged.connect(self._on_end_frac_slider_changed)
        
        self.end_frac_spin = QDoubleSpinBox()
        self.end_frac_spin.setMinimum(0.0)
        self.end_frac_spin.setMaximum(1.0)
        self.end_frac_spin.setSingleStep(0.01)
        self.end_frac_spin.setDecimals(2)
        self.end_frac_spin.setValue(1.0)
        self.end_frac_spin.valueChanged.connect(self._on_end_frac_spin_changed)
        
        end_frac_controls.addWidget(self.end_frac_slider, 3)
        end_frac_controls.addWidget(self.end_frac_spin, 1)
        end_frac_layout.addLayout(end_frac_controls)
        creation_layout.addLayout(end_frac_layout)

        # Add ROI button
        self.add_btn = QPushButton("Add ROI")
        self.add_btn.clicked.connect(self._on_add)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        creation_layout.addWidget(self.add_btn)
        
        # Update ROI button
        self.update_btn = QPushButton("Update ROI")
        self.update_btn.clicked.connect(self._on_update)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        creation_layout.addWidget(self.update_btn)
        
        controls_container.addWidget(creation_group)
        
        # ROI Management Group
        management_group = QGroupBox("ROI List")
        management_group.setStyleSheet("QGroupBox::title { color: black; font-weight: bold; }")
        management_layout = QVBoxLayout()
        management_group.setLayout(management_layout)
        
        # ROI list
        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._on_roi_selected)
        self.roi_list.setMinimumHeight(150)
        management_layout.addWidget(self.roi_list)
        
        # Management buttons
        button_layout = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._on_delete)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.delete_btn)
        
        # Clear all button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._on_clear_all)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        button_layout.addWidget(self.clear_btn)
        management_layout.addLayout(button_layout)
        
        controls_container.addWidget(management_group)
        
        # Add stretch to push controls to top
        controls_container.addStretch()
        
        content_layout.addLayout(controls_container, 1)  # Less space for controls
        layout.addLayout(content_layout)

        # Initial setup
        self._update_buttons()

    def initialize_step(self):
        """Initialize the step when it becomes active"""
        # Sync any existing ROIs from session to local controller
        self.rois.clear()
        self.roi_list.clear()
        
        for name, roi in self.session.rois.items():
            self.rois[name] = roi
            volume_text = f" (Vol: {roi.total_volume}µL)"
            frac_text = f" (Start: {roi.start_frac:.2f}, End: {roi.end_frac:.2f})"
            self.roi_list.addItem(f"{name}{volume_text}{frac_text}")
        
        self._update_frame()
        self._update_buttons()

    def reset_on_new_file(self):
        """Reset all ROIs and steps when a new file is loaded"""
        self.rois.clear()
        self.roi_list.clear()
        self.pending_roi = None
        self.start_frac_slider.setValue(0)
        self.start_frac_spin.setValue(0.0)
        self.end_frac_slider.setValue(100)
        self.end_frac_spin.setValue(1.0)
        self.channel_combo.setCurrentText("auto")
        self._update_buttons()
        logging.info("Reset all ROIs and steps for new file load.")

    def _on_start_frac_slider_changed(self, value):
        """Handle start fraction slider change."""
        fraction = value / 100.0
        self.start_frac_spin.setValue(fraction)

    def _on_start_frac_spin_changed(self, value):
        """Handle start fraction spinbox change."""
        self.start_frac_slider.setValue(int(value * 100))

    def _on_end_frac_slider_changed(self, value):
        """Handle end fraction slider change."""
        fraction = value / 100.0
        self.end_frac_spin.setValue(fraction)

    def _on_end_frac_spin_changed(self, value):
        """Handle end fraction spinbox change."""
        self.end_frac_slider.setValue(int(value * 100))

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
            self.pending_roi = None  # Clear any pending ROI when starting new drawing
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
                # Store the completed rectangle as pending ROI
                if self.start_point and self.current_point:
                    x = min(self.start_point.x(), self.current_point.x())
                    y = min(self.start_point.y(), self.current_point.y())
                    w = abs(self.current_point.x() - self.start_point.x())
                    h = abs(self.current_point.y() - self.start_point.y())
                    self.pending_roi = (int(x), int(y), int(w), int(h))
                self._update_frame()

    def _on_roi_selected(self, current_row):
        """Handle ROI selection in the list"""
        # Update ROI settings when an ROI is selected
        selected_item = self.roi_list.currentItem()
        if selected_item:
            roi_name = selected_item.text().split(' ')[0]  # Extract name from display text
            if roi_name in self.rois:
                roi = self.rois[roi_name]
                self.name_input.setText(roi.name)
                self.volume_input.setText(str(roi.total_volume))
                self.start_frac_spin.setValue(roi.start_frac)
                self.end_frac_spin.setValue(roi.end_frac)
                self.channel_combo.setCurrentText(roi.channel)

        # Update frame to show selection highlighting
        self._update_frame()
        self._update_buttons()

    def _on_add(self):
        """Add a new ROI with the current drawing"""
        if not self.pending_roi:
            QMessageBox.warning(self, "Error", "Please draw an ROI rectangle first by clicking and dragging on the frame.")
            return

        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a name for the ROI before adding it.")
            return

        if name in self.rois:
            QMessageBox.warning(self, "Error", f"An ROI with the name '{name}' already exists. Please choose a different name.")
            return

        volume_text = self.volume_input.text().strip()
        if not volume_text:
            QMessageBox.warning(self, "Error", "Please enter a volume value in µL.")
            return
            
        try:
            volume = float(volume_text)
            if volume <= 0:
                QMessageBox.warning(self, "Error", "Volume must be a positive number.")
                return
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number for the volume.")
            return

        # Use the pending ROI rectangle
        x, y, w, h = self.pending_roi

        try:
            # Get fraction values from UI
            start_frac = self.start_frac_spin.value()
            end_frac = self.end_frac_spin.value()
            
            channel = self.channel_combo.currentText()  # Get selected channel

            # Create ROI with user-specified values
            roi = ROI(
                name=name,
                rect=(int(x), int(y), int(w), int(h)),
                channel=channel,
                total_volume=volume,
                start_frac=start_frac,
                end_frac=end_frac,
                interval=1,
                smoothing_window=5
            )
            self.rois[name] = roi
            self.session.rois[name] = roi  # Also add to session
            
            # Update UI
            volume_text = f" (Vol: {volume}µL)"
            frac_text = f" (Start: {start_frac:.2f}, End: {end_frac:.2f})"
            self.roi_list.addItem(f"{name}{volume_text}{frac_text}")
            self.name_input.clear()
            self.volume_input.clear()
            # Reset fraction controls to defaults
            self.start_frac_spin.setValue(0.0)
            self.start_frac_slider.setValue(0)
            self.end_frac_spin.setValue(1.0)
            self.end_frac_slider.setValue(100)
            self.start_point = None
            self.current_point = None
            self.pending_roi = None  # Clear the pending ROI after adding
            
            # Update button states
            self._update_buttons()
            
            self._update_frame()

            # Notify UIController to enable the next button
            if hasattr(self.ui_controller, 'update_next_button_state'):
                self.ui_controller.update_next_button_state(len(self.rois) > 0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create ROI: {str(e)}")

    def _on_update(self):
        """Update the selected ROI with new settings"""
        current_row = self.roi_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Error", "Please select an ROI to update.")
            return

        item = self.roi_list.item(current_row)
        roi_name = item.text().split(" ")[0]  # Extract name from display text

        if roi_name not in self.rois:
            QMessageBox.warning(self, "Error", "Selected ROI not found.")
            return

        roi = self.rois[roi_name]

        # Update ROI settings
        try:
            volume_text = self.volume_input.text().strip()
            if volume_text:
                roi.total_volume = float(volume_text)

            roi.start_frac = self.start_frac_spin.value()
            roi.end_frac = self.end_frac_spin.value()

            channel = self.channel_combo.currentText()  # Get selected channel
            roi.channel = channel

            self.rois[roi_name] = roi
            self.session.rois[roi_name] = roi

            # Update display text
            volume_text = f" (Vol: {roi.total_volume}µL)"
            frac_text = f" (Start: {roi.start_frac:.2f}, End: {roi.end_frac:.2f})"
            item.setText(f"{roi_name}{volume_text}{frac_text}")

            QMessageBox.information(self, "Success", "ROI updated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update ROI: {str(e)}")

    def _on_delete(self):
        """Delete the selected ROI"""
        current_row = self.roi_list.currentRow()
        if current_row >= 0:
            # Remove item and extract ROI name (text formatted as 'Name (Vol:...')
            item = self.roi_list.takeItem(current_row)
            if item:
                text = item.text()
                roi_name = text.split(' (')[0]
                # Delete from local and session rois
                if roi_name in self.rois:
                    del self.rois[roi_name]
                if hasattr(self.session, 'rois') and roi_name in self.session.rois:
                    del self.session.rois[roi_name]
                # Refresh display
                self._update_frame()
                self._update_buttons()
        else:
            QMessageBox.information(self, "No Selection", "Please select an ROI from the list to delete.")

    def _on_clear_all(self):
        """Clear all ROIs"""
        if not self.rois:
            QMessageBox.information(self, "No ROIs", "There are no ROIs to clear.")
            return
            
        reply = QMessageBox.question(
            self, 
            "Clear All ROIs", 
            f"Are you sure you want to delete all {len(self.rois)} ROIs? This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.roi_list.clear()
            self.rois.clear()
            self.session.rois.clear()
            self.pending_roi = None  # Clear any pending ROI as well
            self._update_frame()
            self._update_buttons()

    def _update_buttons(self):
        """Update button states based on current ROIs"""
        has_rois = len(self.rois) > 0
        has_selection = self.roi_list.currentRow() >= 0
        
        self.delete_btn.setEnabled(has_selection)
        self.clear_btn.setEnabled(has_rois)

    def _on_confirm(self):
        self.session.rois = self.rois
        self.on_complete()

    def _update_frame(self):
        """Update the frame display with current video frame and ROIs"""
        # Check if video is loaded and session has a start frame
        if (not hasattr(self.engine, 'cap') or 
            not self.engine.cap or 
            not hasattr(self.session, 'start_frame') or 
            self.session.start_frame is None):
            # Show placeholder text when no video is loaded
            self.frame_label.setText("No video loaded.\nPlease load a video first.")
            self.frame_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; color: #666; font-size: 14pt;")
            return

        # Get the frame
        frame = self.engine.get_frame(self.session.start_frame)
        if frame is None:
            self.frame_label.setText("Unable to load frame.\nPlease check video file.")
            self.frame_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; color: #666; font-size: 14pt;")
            return

        # Apply rotation if it exists in the session
        if hasattr(self.session, 'rotation_matrix') and self.session.rotation_matrix is not None:
            h, w = frame.shape[:2]
            # Calculate new image dimensions after rotation
            M = self.session.rotation_matrix
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Apply rotation
            frame = cv2.warpAffine(frame, M, (new_w, new_h))

        # Convert frame to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Create a painter for drawing
        painter = QPainter(pixmap)
        
        # Draw existing ROIs
        selected_roi_name = None
        selected_item = self.roi_list.currentItem()
        if selected_item:
            selected_roi_name = selected_item.text().split(' ')[0]
        
        for name, roi in self.rois.items():
            # Use green for selected ROI, red for others
            if name == selected_roi_name:
                pen = QPen(QColor(0, 255, 0))  # Green for selected ROI
                pen.setWidth(6)  # Thicker for selected
            else:
                pen = QPen(QColor(255, 0, 0))  # Red for unselected ROIs
                pen.setWidth(4)
            
            painter.setPen(pen)
            x, y, w, h = roi.rect
            painter.drawRect(x, y, w, h)
            # Draw name above the ROI
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(x, y - 5, name)

        # Draw current ROI if drawing
        if self.drawing and self.start_point and self.current_point:
            pen = QPen(QColor(0, 255, 0))  # Green for active drawing
            pen.setWidth(4)  # Increased thickness from 2 to 4
            painter.setPen(pen)
            x = min(self.start_point.x(), self.current_point.x())
            y = min(self.start_point.y(), self.current_point.y())
            w = abs(self.current_point.x() - self.start_point.x())
            h = abs(self.current_point.y() - self.start_point.y())
            painter.drawRect(int(x), int(y), int(w), int(h))
        
        # Draw pending ROI (after drawing is complete but before adding)
        elif self.pending_roi:
            pen = QPen(QColor(0, 150, 255))  # Blue for pending ROI
            pen.setWidth(5)  # Increased thickness from 3 to 5
            painter.setPen(pen)
            x, y, w, h = self.pending_roi
            painter.drawRect(x, y, w, h)
            # Add text indicator
            font = painter.font()
            font.setPointSize(10)
            font.setWeight(QFont.Bold)
            painter.setFont(font)
            painter.drawText(x, y - 8, "New ROI - Enter name and click 'Add ROI'")

        painter.end()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.frame_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled_pixmap)
        self.frame_label.setAlignment(Qt.AlignCenter)

        # Make Next button always available
        self.next_btn.setEnabled(True)

    def update_roi(self, roi_name, new_name):
        if roi_name in self.rois and new_name != roi_name:
            # Update the ROI object
            roi = self.rois[roi_name]
            roi.name = new_name
            
            # Update the dictionary key
            del self.rois[roi_name]
            self.rois[new_name] = roi
            # Update session rois if they exist
            if hasattr(self.session, 'rois') and roi_name in self.session.rois:
                del self.session.rois[roi_name]
                self.session.rois[new_name] = roi
            # Refresh list and preview
            self.roi_list.clear()
            for name, r in self.rois.items():
                vol_text = f" (Vol: {r.total_volume}µL)"
                frac_text = f" (Start: {r.start_frac:.2f}, End: {r.end_frac:.2f})"
                self.roi_list.addItem(f"{name}{vol_text}{frac_text}")
            self._update_frame()
            self._update_buttons()
        else:
            # Copy, rename, and delete original ROI
            if roi_name in self.rois:
                roi = self.rois[roi_name]
                new_roi = roi.copy()
                new_roi.name = new_name
                self.rois[new_name] = new_roi
                del self.rois[roi_name]
                if hasattr(self.session, 'rois') and roi_name in self.session.rois:
                    self.session.rois[new_name] = new_roi
                    del self.session.rois[roi_name]
                self.redraw_preview()
