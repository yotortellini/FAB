# controllers/analysis_controller.py

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QSpinBox, QProgressBar, QFileDialog,
                            QMessageBox, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import rcParams
import logging

from models.video_session import VideoSession
from models.analysis_results import AnalysisResults
from engine.video_engine import VideoEngine
from helpers.auto_channel import detect_best_channel

class AnalysisWorker(QThread):
    """Worker thread for running analysis"""
    progress = pyqtSignal(str, int, int, int)  # ROI name, roi_progress, current_roi_num, total_rois
    finished = pyqtSignal(dict)  # Results dictionary
    error = pyqtSignal(str)

    def __init__(self, session, engine, smooth_window, sampling_freq):
        super().__init__()
        self.session = session
        self.engine = engine
        self.smooth_window = smooth_window
        self.sampling_freq = sampling_freq
        self.results = {}

    def run(self):
        try:
            total_rois = len(self.session.rois)
            roi_num = 0
            
            for name, roi in self.session.rois.items():
                roi_num += 1
                self.progress.emit(name, 0, roi_num, total_rois)
                samples = []
                
                # Calculate frame range based on ROI start/end fractions
                total_frames = self.session.end_frame - self.session.start_frame
                start_frame = self.session.start_frame + int(total_frames * roi.start_frac)
                end_frame = self.session.start_frame + int(total_frames * roi.end_frac)
                
                # Ensure start_frame aligns with start_frac
                if roi.start_frac == 0:
                    start_frame = self.session.start_frame
                
                # Calculate frame step based on sampling frequency
                video_fps = self.session.fps
                frame_step = max(1, int(video_fps / self.sampling_freq))
                if frame_step > total_frames:
                    logging.warning(f"Sampling rate exceeds total frames for ROI {name}. Adjusting frame_step to 1.")
                    frame_step = 1
                frame_range = range(start_frame, end_frame, frame_step)
                
                # Debug: Log frame range calculation
                logging.info(f"ROI {name}: total_frames={total_frames}, start_frame={start_frame}, end_frame={end_frame}")
                logging.info(f"ROI {name}: frame_step={frame_step}, frame_range length={len(list(frame_range))}")
                
                # Debug: Log first few intensity samples
                first_few_samples = []
                
                for i, idx in enumerate(frame_range):
                    frame = self.engine.get_frame(idx)
                    if frame is None:
                        continue

                    x, y, w, h = roi.rect
                    roi_pixels = frame[y:y+h, x:x+w]

                    # Extract the specified channel
                    if roi.channel == 'auto':
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.information(None, "Auto-Channel Selection", "Auto-Channel Selection is currently under development.")
                        roi.channel = 'R'  # Default to 'R' for now

                    if roi.channel == 'R':
                        intensity = cv2.mean(roi_pixels[:,:,2])[0]  # OpenCV uses BGR
                    elif roi.channel == 'G':
                        intensity = cv2.mean(roi_pixels[:,:,1])[0]
                    elif roi.channel == 'B':
                        intensity = cv2.mean(roi_pixels[:,:,0])[0]
                    else:
                        intensity = cv2.mean(roi_pixels)[0]  # Default to auto

                    samples.append(float(intensity))
                    
                    # Debug: Store first 5 samples for logging
                    if len(first_few_samples) < 5:
                        first_few_samples.append(float(intensity))
                    
                    roi_progress = int((i+1)/len(frame_range)*100)
                    self.progress.emit(name, roi_progress, roi_num, total_rois)

                # Debug: Log raw intensity samples
                volume_raw = np.array(samples)
                logging.info(f"ROI {name}: raw intensity samples (first 5): {volume_raw[:5]}")

                # Normalize raw intensities to 0-1 and scale to ROI total_volume
                ptp = volume_raw.ptp()
                if ptp > 0:
                    norm = (volume_raw - volume_raw.min()) / ptp
                else:
                    norm = np.zeros_like(volume_raw)
                volume = norm * roi.total_volume
                # Debug: Log normalized volume samples
                logging.info(f"ROI {name}: normalized volume (first 5): {volume[:5]}")

                # Calculate time array like the original: frame_index * interval / fps
                # Use actual sampling frequency for time calculation
                actual_frame_step = max(1, int(video_fps / self.sampling_freq))
                times = np.arange(len(volume)) * actual_frame_step / video_fps
                
                # Apply time multiplier if present (from original logic)
                if hasattr(self.session, 'time_multiplier'):
                    times = times * self.session.time_multiplier
                else:
                    logging.info("No time multiplier found. Using default time calculation.")
                    
                # Calculate flow rate and apply smoothing if requested
                flow = np.gradient(volume, times)  # Now in µL/s
                if self.smooth_window > 1:
                    flow = np.convolve(flow, np.ones(self.smooth_window)/self.smooth_window, mode='same')
                
                self.results[name] = (times, volume, flow)
                
                # Emit completion for this ROI
                self.progress.emit(name, 100, roi_num, total_rois)
            
            self.finished.emit(self.results)
        except Exception as e:
            self.error.emit(str(e))

class AnalysisController(QWidget):
    """Controller for Step 5: Analysis"""

    def __init__(self, parent, session, engine, on_complete=None, ui_controller=None):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)
        self.ui_controller = ui_controller
        self.results = {}
        self.plot_lines = {}  # Store plot line references for highlighting
        # Prepare a color cycle for ROI plots
        self.colors = rcParams['axes.prop_cycle'].by_key()['color']
        
        self._build_ui()
        self._setup_plots()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Step 5: Analysis")
        title.setStyleSheet("font-size: 16pt;")
        layout.addWidget(title)

        # Progress bar and status
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Controls
        controls = QHBoxLayout()
        
        # Smoothing window
        controls.addWidget(QLabel("Smoothing Window:"))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 20)
        self.smooth_spin.setValue(1)
        controls.addWidget(self.smooth_spin)

        # Sampling frequency
        controls.addWidget(QLabel("Sampling Frequency (Hz):"))
        self.sampling_freq_spin = QSpinBox()
        self.sampling_freq_spin.setRange(1, 1000)
        self.sampling_freq_spin.setValue(30)  # Default 30 Hz
        self.sampling_freq_spin.setToolTip("Frames per second for analysis sampling")
        controls.addWidget(self.sampling_freq_spin)

        # Run button
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._on_run)
        controls.addWidget(self.run_btn)

        layout.addLayout(controls)

        # Plots
        plots_layout = QHBoxLayout()
        self.canvas_vol = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        self.canvas_flow = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        plots_layout.addWidget(self.canvas_vol)
        plots_layout.addWidget(self.canvas_flow)
        layout.addLayout(plots_layout)

        # ROI List
        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._on_roi_selected)
        layout.addWidget(self.roi_list)

    def _setup_plots(self):
        """Set up plots and ROI information display"""
        # Setup volume plot
        self.canvas_vol.figure.clear()
        self.ax_vol = self.canvas_vol.figure.add_subplot(111)
        self.ax_vol.set_title("Volume vs Time")
        self.ax_vol.set_xlabel("Time (s)")
        self.ax_vol.set_ylabel("Volume (µL)")

        # Setup flow plot
        self.canvas_flow.figure.clear()
        self.ax_flow = self.canvas_flow.figure.add_subplot(111)
        self.ax_flow.set_title("Flow Rate vs Time")
        self.ax_flow.set_xlabel("Time (s)")
        self.ax_flow.set_ylabel("Flow Rate (µL/s)")

        # Ensure distinct axes for each plot
        self.canvas_vol.figure.tight_layout()
        self.canvas_flow.figure.tight_layout()

        # Populate ROI list with information
        self.roi_list.clear()
        for name, roi in self.session.rois.items():
            channel_info = f"Channel: {roi.channel}"
            roi_info = f"ROI: {name}, {channel_info}, Start: {roi.start_frac:.2f}, End: {roi.end_frac:.2f}, Volume: {roi.total_volume:.2f} µL"
            self.roi_list.addItem(roi_info)

    def _on_run(self):
        # Check if there are ROIs to analyze
        if not hasattr(self.session, 'rois') or not self.session.rois:
            QMessageBox.warning(self, "No ROIs", "Please define ROI regions before running analysis.")
            return
            
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status_label.setText(f"Starting analysis of {len(self.session.rois)} ROI(s)...")
        
        # Clear previous plots
        self.ax_vol.clear()
        self.ax_flow.clear()
        self._setup_plots()
        
        # Start analysis in worker thread
        self.worker = AnalysisWorker(self.session, self.engine, self.smooth_spin.value(), self.sampling_freq_spin.value())
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_analysis_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _update_progress(self, roi_name, roi_progress, current_roi, total_rois):
        # Calculate overall progress
        roi_weight = 100 / total_rois  # Each ROI is worth this much of total progress
        completed_rois_progress = (current_roi - 1) * roi_weight
        current_roi_progress = (roi_progress / 100) * roi_weight
        overall_progress = int(completed_rois_progress + current_roi_progress)
        
        self.status_label.setText(f"Analyzing ROI {current_roi}/{total_rois}: {roi_name} ({roi_progress}%)")
        self.progress.setValue(overall_progress)

    def _on_analysis_complete(self, results):
        """Handle completion of analysis"""
        self.results = results
        self.status_label.setText(f"Analysis complete! Processed {len(results)} ROIs.")
        self.progress.setValue(100)
        self.progress.hide()

        # Re-enable buttons
        self.run_btn.setEnabled(True)

        # Plot results
        plot_count = 0
        self.plot_lines = {}  # Reset plot line references
        # Plot each ROI with a unique color
        for idx, (name, (times, volume, flow)) in enumerate(results.items()):
            # Ensure volume calculation matches original logic
            volume = np.array(volume)

            # Cycle through predefined colors
            color = self.colors[idx % len(self.colors)]
            vol_line, = self.ax_vol.plot(times, volume, label=name, color=color)
            flow_line, = self.ax_flow.plot(times, flow, label=name, color=color)
            
            # Store line references
            self.plot_lines[name] = {'volume': vol_line, 'flow': flow_line}
            
            plot_count += 1

        # Only add legends if there are plots
        if plot_count > 0:
            self.ax_vol.legend()
            self.ax_flow.legend()

        self.canvas_vol.draw()
        self.canvas_flow.draw()

        # Store results in session
        self.session.analysis_results = {
            name: AnalysisResults(time=t, volume=v, flow=f, 
                                  smoothing_window=self.smooth_spin.value(),
                                  sampling_frequency=self.sampling_freq_spin.value())
            for name, (t, v, f) in results.items()
        }
        
        # Notify UI controller that analysis is complete
        if self.ui_controller and hasattr(self.ui_controller, 'on_analysis_complete'):
            self.ui_controller.on_analysis_complete()

    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.run_btn.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Analysis failed.")

    def _on_save(self):
        # Get base filename from user
        base_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "All Files (*.*)"
        )
        
        if not base_name:
            return

        try:
            # Save CSV results
            with open(f"{base_name}_results.csv", 'w') as f:
                # Write header
                f.write("Time,")
                for name in self.results.keys():
                    f.write(f"{name}_Volume,{name}_Flow,")
                f.write("\n")
                
                # Write data
                first_result = next(iter(self.results.values()))
                for i in range(len(first_result[0])):
                    f.write(f"{first_result[0][i]},")
                    for name, (_, vol, flow) in self.results.items():
                        f.write(f"{vol[i]},{flow[i]},")
                    f.write("\n")

            # Ensure all lines are uniform (reset any highlights)
            for lines in self.plot_lines.values():
                lines['volume'].set_linewidth(1)
                lines['flow'].set_linewidth(1)
            # Redraw canvases before saving to capture reset state
            self.canvas_vol.draw()
            self.canvas_flow.draw()
            # Save plots
            self.canvas_vol.figure.savefig(f"{base_name}_volume.png")
            self.canvas_flow.figure.savefig(f"{base_name}_flow.png")
            
            # Save ROI overlay image
            self._save_roi_overlay(f"{base_name}_roi_overlay.png")

            QMessageBox.information(self, "Save Complete", 
                                  f"Results saved as:\n{base_name}_results.csv\n"
                                  f"{base_name}_volume.png\n{base_name}_flow.png\n"
                                  f"{base_name}_roi_overlay.png")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _save_roi_overlay(self, filename):
        """Save a side-by-side overlay of start and end frames with ROI information"""
        try:
            start_frame = self.engine.get_frame(self.session.start_frame).copy()
            end_frame = self.engine.get_frame(self.session.end_frame).copy()

            # Burn ROI rectangles and names into each frame BEFORE composing
            for name, roi in self.session.rois.items():
                x, y, w, h = roi.rect
                color = (0, 255, 0)  # Green for ROI
                
                # Draw on start frame
                cv2.rectangle(start_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(start_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw on end frame
                cv2.rectangle(end_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(end_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Create side-by-side image
            h, w = start_frame.shape[:2]
            combined_image = np.zeros((h, w * 2, 3), dtype=start_frame.dtype)
            combined_image[:, :w] = start_frame
            combined_image[:, w:] = end_frame

            # Add labels for start and end frames
            cv2.putText(combined_image, "Start Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(combined_image, "End Frame", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Save the combined image
            cv2.imwrite(filename, combined_image)
            logging.info(f"Saved ROI overlay to {filename}")
        except Exception as e:
            logging.error(f"Failed to save ROI overlay: {e}")

    def initialize_step(self):
        """Initialize the step when it becomes active"""
        # Set sampling frequency to video FPS if available
        if hasattr(self.session, 'fps') and self.session.fps:
            self.sampling_freq_spin.setValue(min(int(self.session.fps), 1000))
        self._update_status()

    def _update_status(self):
        status = "Ready"
        if hasattr(self.session, 'analysis_results') and self.session.analysis_results:
            status = f"Loaded results for {len(self.session.analysis_results)} ROIs"
        self.status_label.setText(status)

        # Debugging: Log ROI parameters
        for name, roi in self.session.rois.items():
            logging.info(f"ROI {name}: start_frac={roi.start_frac}, end_frac={roi.end_frac}, total_volume={roi.total_volume}")

    def _on_roi_selected(self, current_row):
        """Handle ROI selection to highlight corresponding plot lines"""
        # Reset all lines to normal weight
        for roi_name, lines in self.plot_lines.items():
            lines['volume'].set_linewidth(1)
            lines['flow'].set_linewidth(1)
        
        # Highlight selected ROI lines
        if current_row >= 0:
            selected_item = self.roi_list.item(current_row)
            if selected_item:
                # Extract ROI name from the list item text, formatted as 'ROI: name, ...'
                text = selected_item.text()
                if text.startswith('ROI: '):
                    roi_name = text.split('ROI: ')[1].split(',')[0]
                else:
                    roi_name = text.split(',')[0]
                # Highlight selected ROI plot lines
                if roi_name in self.plot_lines:
                    self.plot_lines[roi_name]['volume'].set_linewidth(3)
                    self.plot_lines[roi_name]['flow'].set_linewidth(3)
        
        # Refresh plots
        self.canvas_vol.draw()
        self.canvas_flow.draw()
