# controllers/analysis_controller.py

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QSpinBox, QProgressBar, QFileDialog,
                            QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from models.video_session import VideoSession
from models.analysis_results import AnalysisResults
from engine.video_engine import VideoEngine

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
                
                # Calculate frame step based on sampling frequency
                video_fps = self.session.fps
                frame_step = max(1, int(video_fps / self.sampling_freq))
                frame_range = range(start_frame, end_frame, frame_step)
                
                for i, idx in enumerate(frame_range):
                    frame = self.engine.get_frame(idx)
                    if frame is None:
                        continue
                        
                    x, y, w, h = roi.rect
                    roi_pixels = frame[y:y+h, x:x+w]
                    
                    # Extract the specified channel
                    if roi.channel == 'auto':
                        intensity = cv2.mean(roi_pixels)[0]  # Average of all channels
                    elif roi.channel == 'R':
                        intensity = cv2.mean(roi_pixels[:,:,2])[0]  # OpenCV uses BGR
                    elif roi.channel == 'G':
                        intensity = cv2.mean(roi_pixels[:,:,1])[0]
                    elif roi.channel == 'B':
                        intensity = cv2.mean(roi_pixels[:,:,0])[0]
                    else:
                        intensity = cv2.mean(roi_pixels)[0]  # Default to auto
                    
                    samples.append(float(intensity))
                    roi_progress = int((i+1)/len(frame_range)*100)
                    self.progress.emit(name, roi_progress, roi_num, total_rois)

                volume = np.array(samples)
                if self.smooth_window > 1:
                    volume = np.convolve(volume, 
                                       np.ones(self.smooth_window)/self.smooth_window, 
                                       mode='valid')

                # Use actual sampling frequency for time calculation
                times = np.arange(len(volume)) / self.sampling_freq
                flow = np.gradient(volume, times)
                
                self.results[name] = (times, volume, flow)
                
                # Emit completion for this ROI
                self.progress.emit(name, 100, roi_num, total_rois)
            
            self.finished.emit(self.results)
        except Exception as e:
            self.error.emit(str(e))

class AnalysisController(QWidget):
    """Controller for Step 5: Analysis"""

    def __init__(self, parent, session, engine, on_complete=None):
        super().__init__(parent)
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)
        self.results = {}
        
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

        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        controls.addWidget(self.save_btn)

        layout.addLayout(controls)

        # Plots
        plots_layout = QHBoxLayout()
        self.canvas_vol = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        self.canvas_flow = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        plots_layout.addWidget(self.canvas_vol)
        plots_layout.addWidget(self.canvas_flow)
        layout.addLayout(plots_layout)

    def _setup_plots(self):
        self.ax_vol = self.canvas_vol.figure.add_subplot(111)
        self.ax_vol.set_title("Volume vs Time")
        self.ax_vol.set_xlabel("Time (s)")
        self.ax_vol.set_ylabel("Volume (a.u.)")

        self.ax_flow = self.canvas_flow.figure.add_subplot(111)
        self.ax_flow.set_title("Flow vs Time")
        self.ax_flow.set_xlabel("Time (s)")
        self.ax_flow.set_ylabel("Flow (a.u.)")

    def _on_run(self):
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status_label.setText("Starting analysis...")
        
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
        self.results = results
        self.status_label.setText(f"Analysis complete! Processed {len(results)} ROIs.")
        self.progress.setValue(100)
        self.progress.hide()
        
        # Re-enable buttons
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Plot results
        plot_count = 0
        for name, (times, volume, flow) in results.items():
            self.ax_vol.plot(times, volume, label=name)
            self.ax_flow.plot(times, flow, label=name)
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
        
        self.save_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(False)  # Don't enable save if there was an error
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

            # Save plots
            self.canvas_vol.figure.savefig(f"{base_name}_volume.png")
            self.canvas_flow.figure.savefig(f"{base_name}_flow.png")

            QMessageBox.information(self, "Save Complete", 
                                  f"Results saved as:\n{base_name}_results.csv\n"
                                  f"{base_name}_volume.png\n{base_name}_flow.png")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

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
