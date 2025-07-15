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
    progress = pyqtSignal(str, int)  # ROI name, progress value
    finished = pyqtSignal(dict)  # Results dictionary
    error = pyqtSignal(str)

    def __init__(self, session, engine, smooth_window):
        super().__init__()
        self.session = session
        self.engine = engine
        self.smooth_window = smooth_window
        self.results = {}

    def run(self):
        try:
            for name, roi in self.session.rois.items():
                self.progress.emit(name, 0)
                samples = []
                frame_range = range(self.session.start_frame, self.session.end_frame)
                
                for i, idx in enumerate(frame_range):
                    frame = self.engine.get_frame(idx)
                    x, y, w, h = roi.bounds
                    roi_pixels = frame[y:y+h, x:x+w]
                    samples.append(float(cv2.mean(roi_pixels)[0]))
                    self.progress.emit(name, int((i+1)/len(frame_range)*100))

                volume = np.array(samples)
                if self.smooth_window > 1:
                    volume = np.convolve(volume, 
                                       np.ones(self.smooth_window)/self.smooth_window, 
                                       mode='valid')

                times = np.arange(len(volume)) / self.session.fps
                flow = np.gradient(volume, times)
                
                self.results[name] = (times, volume, flow)
            
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
        self.progress.setValue(0)
        self.progress.show()
        
        # Clear previous plots
        self.ax_vol.clear()
        self.ax_flow.clear()
        self._setup_plots()
        
        # Start analysis in worker thread
        self.worker = AnalysisWorker(self.session, self.engine, self.smooth_spin.value())
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_analysis_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _update_progress(self, roi_name, value):
        self.status_label.setText(f"Analyzing: {roi_name}")
        self.progress.setValue(value)

    def _on_analysis_complete(self, results):
        self.results = results
        self.status_label.clear()
        self.progress.hide()
        
        # Plot results
        for name, (times, volume, flow) in results.items():
            self.ax_vol.plot(times, volume, label=name)
            self.ax_flow.plot(times, flow, label=name)
        
        self.ax_vol.legend()
        self.ax_flow.legend()
        self.canvas_vol.draw()
        self.canvas_flow.draw()
        
        # Store results in session
        self.session.analysis_results = {
            name: AnalysisResults(time=t, volume=v, flow=f, 
                                smoothing_window=self.smooth_spin.value())
            for name, (t, v, f) in results.items()
        }
        
        self.save_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.run_btn.setEnabled(True)
        self.progress.hide()
        self.status_label.clear()

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
