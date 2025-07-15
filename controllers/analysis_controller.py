# controllers/analysis_controller.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional
import numpy as np
import os, csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFont
import tkinter.simpledialog
import threading

from models.video_session import VideoSession
from models.analysis_results import AnalysisResults
from engine.video_engine import VideoEngine
from controllers.lighting_correction_controller import LightingCorrectionController

"""
analysis_controller.py

Provides the AnalysisController class for running and displaying
analysis of regions of interest (ROIs) on video frames, including
volume and flow rate plotting, result confirmation, and saving.
"""

class AnalysisController:
    """
    Controller for Step 6: Analysis.

    Builds the UI for analysis controls, runs analysis on all defined ROIs,
    displays volume and flow plots, and handles saving and confirmation of results.
    """
    def __init__(self,
                 parent: tk.Frame,
                 session: VideoSession,
                 engine: VideoEngine,
                 on_complete: Optional[Callable[[], None]] = None,
                 on_edit_rois: Optional[Callable[[], None]] = None):
        """
        Initialize the AnalysisController.

        Args:
            parent (tk.Frame): Parent frame for UI components.
            session (VideoSession): Video session containing video and ROI data.
            engine (VideoEngine): Video engine for frame retrieval and processing.
            on_complete (Optional[Callable[[], None]]): Callback when analysis completes.
            on_edit_rois (Optional[Callable[[], None]]): Callback to trigger ROI editing.
        """
        self.parent       = parent
        self.session      = session
        self.engine       = engine
        self.on_complete  = on_complete or (lambda: None)
        self.on_edit_rois = on_edit_rois or (lambda: None)

        self.results = {}
        self._build_ui()

    def _build_ui(self):
        """
        Construct the analysis UI components.

        Creates labels, buttons, spinboxes, progress bar, and plot canvases
        for user interaction during analysis.
        """
        ttk.Label(self.parent, text="Step 6: Analysis", font=(None,16)).pack(pady=10)

        # Controls...
        ctrl = ttk.Frame(self.parent)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="Frame Interval:").pack(side=tk.LEFT)
        self.interval_var = tk.IntVar(value=5)
        ttk.Spinbox(ctrl, from_=1, to=100, width=5,
                    textvariable=self.interval_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(ctrl, text="Smoothing Window:").pack(side=tk.LEFT, padx=(20,0))
        self.smooth_var = tk.IntVar(value=2)
        ttk.Spinbox(ctrl, from_=1, to=50, width=5,
                    textvariable=self.smooth_var).pack(side=tk.LEFT, padx=5)
        self.interval_var.trace_add('write', lambda *_: self._mark_dirty())
        self.smooth_var.trace_add('write',   lambda *_: self._mark_dirty())

        # Run & Progress
        self.run_btn = ttk.Button(self.parent, text="Run Analysis", command=self._on_run)
        self.run_btn.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(self.parent, text="")
        self.status_label.pack()

        # Progress bar
        self.prog = ttk.Progressbar(self.parent, mode='determinate')
        self.prog.pack(fill=tk.X, padx=20, pady=(0,10))

        # Lighting, Edit ROIs & Save
        bar = ttk.Frame(self.parent)
        bar.pack(fill=tk.X, padx=10, pady=(0,10))
        ttk.Button(bar, text="Lighting Correction…", command=self._open_lighting).pack(side=tk.LEFT)
        ttk.Button(bar, text="Edit ROIs", command=self.on_edit_rois).pack(side=tk.LEFT, padx=10)
        self.save_btn = ttk.Button(bar, text="Save Results…", state="disabled", command=self._on_save)
        self.save_btn.pack(side=tk.LEFT, padx=10)

        # Volume plot (thin vertically, no labels yet)
        fig_v = Figure(figsize=(5,2.5))
        self.ax_vol = fig_v.add_subplot(111)
        self.ax_vol.set_xlabel("")
        self.ax_vol.set_ylabel("")
        self.canvas_vol = FigureCanvasTkAgg(fig_v, master=self.parent)
        self.canvas_vol.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Flow plot
        fig_f = Figure(figsize=(5,2.5))
        self.ax_flow = fig_f.add_subplot(111)
        self.ax_flow.set_xlabel("")
        self.ax_flow.set_ylabel("")
        self.canvas_flow = FigureCanvasTkAgg(fig_f, master=self.parent)
        self.canvas_flow.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _mark_dirty(self):
        """Mark analysis state as dirty, enabling rerun and disabling save/confirm."""
        self.run_btn.config(state="normal")
        self.save_btn.config(state="disabled")

    def _open_lighting(self):
        """Open the lighting correction UI and apply correction upon completion."""
        LightingCorrectionController(
            parent=self.parent,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: messagebox.showinfo("Lighting","Correction applied.")
        )

    def _on_run(self):
        """Start analysis in a background thread to keep UI responsive."""
        self.run_btn.config(state="disabled")
        thread = threading.Thread(target=self._run_analysis)
        thread.start()

    def _run_analysis(self):
        """Actual analysis logic (moved from _on_run)."""
        rois = self.session.rois
        if not rois:
            self.parent.after(0, lambda: messagebox.showerror("Error","No ROIs defined."))
            self.parent.after(0, lambda: self.run_btn.config(state="normal"))
            return

        # Determine frames to sample based on interval
        iv = self.interval_var.get()
        frame_range = range(self.session.start_frame, self.session.end_frame, iv)
        steps = len(frame_range)
        self.prog.config(maximum=steps, value=0)

        # Reset previous analysis results and plots
        self.results.clear()
        self.ax_vol.clear()
        self.ax_flow.clear()
        self.status_label.config(text="")

        # Compute time array for plotting
        base_times = np.arange(steps) * iv / self.session.fps
        times = base_times * getattr(self.session, 'time_multiplier', 1.0)

        # Loop over each ROI to sample pixel intensities
        for name, roi in rois.items():
            # update status
            self.parent.after(0, lambda name=name: self.status_label.config(text=f"Analyzing: {name}"))
            samples = []
            for idx in frame_range:
                # Retrieve and optionally rotate frame
                frame = self.engine.get_frame(idx)
                if getattr(self.session, 'rotation_matrix', None) is not None:
                    frame = self.engine.apply_rotation(frame, self.session.rotation_matrix)
                x,y,w,h = roi.rect
                patch = frame[y:y+h, x:x+w]
                ch = {'B':0,'G':1,'R':2}[roi.channel]
                samples.append(patch[:,:,ch].mean())
                self.parent.after(0, self.prog.step, 1)
                self.parent.after(0, self.parent.update_idletasks)

            arr = np.array(samples)
            # Normalize intensity to volume
            norm   = (arr - arr.min()) / (arr.ptp() or 1)
            volume = norm * roi.total_volume
            # Compute flow rate as time derivative of volume
            flow   = np.gradient(volume, times)

            self.results[name] = (times, volume, flow)

            # Apply smoothing and update plots
            w = max(1, self.smooth_var.get())
            vol_s  = np.convolve(volume, np.ones(w)/w, mode='same')
            flow_s = np.convolve(flow,   np.ones(w)/w, mode='same')
            self.ax_vol.plot(times, vol_s, label=name)
            self.ax_flow.plot(times, flow_s, label=name)
            self.parent.after(0, self.canvas_vol.draw)
            self.parent.after(0, self.canvas_flow.draw)
            self.parent.after(0, lambda: self.prog.config(value=0))

        # Finalize plot labels, legends, and redraw canvases
        self.ax_vol.set_title("Volume vs Time")
        self.ax_vol.set_xlabel("Time (s)")
        self.ax_vol.set_ylabel("Volume (µL)")
        self.ax_vol.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)
        self.ax_vol.legend()

        self.ax_flow.set_title("Flow Rate vs Time")
        self.ax_flow.set_xlabel("Time (s)")
        self.ax_flow.set_ylabel("Flow Rate (µL/s)")
        self.ax_flow.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)
        self.ax_flow.legend()

        self.canvas_vol.draw()
        self.canvas_flow.draw()

        # clear status text now that we're done
        self.parent.after(0, lambda: self.status_label.config(text=""))

        # Store results in session immediately after analysis
        self.session.analysis_results = {
            name: AnalysisResults(time=t, volume=v, flow=f, smoothing_window=self.smooth_var.get())
            for name, (t, v, f) in self.results.items()
        }

        # enable save
        self.parent.after(0, lambda: self.save_btn.config(state="normal"))

    def _on_save(self):
        """
        Save analysis data and plots to a user-selected folder.

        - Prompts for output directory.
        - Exports CSV with time and volume data.
        - Saves volume and flow plot images.
        - Generates and saves an ROI overlay image.
        """
        # Prompt for a base file name
        base_name = tkinter.simpledialog.askstring("Save Results", "Enter a base file name for all saved assets:")
        if not base_name:
            return  # User cancelled

        # Prompt for a directory to save files
        directory = filedialog.askdirectory(title="Select Folder to Save Results")
        if not directory:
            return  # User cancelled

        # Build full file paths
        results_path = f"{directory}/{base_name}_results.csv"
        volume_plot_path = f"{directory}/{base_name}_volume.png"
        flow_plot_path = f"{directory}/{base_name}_flow.png"

        # Save results (implement your actual saving logic here)
        self._save_results_csv(results_path)
        self._save_plot(self.ax_vol, volume_plot_path)
        self._save_plot(self.ax_flow, flow_plot_path)
        messagebox.showinfo(
            "Save Complete",
            f"Results saved as:\n{results_path}\n{volume_plot_path}\n{flow_plot_path}"
        )

    def _save_results_csv(self, path):
        """
        Save analysis results to a CSV file.
        Each ROI's results are written as separate blocks with headers.
        """
        if not hasattr(self.session, "analysis_results") or not self.session.analysis_results:
            messagebox.showerror("Save Error", "No analysis results to save.")
            return

        try:
            with open(path, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                for roi_name, result in self.session.analysis_results.items():
                    writer.writerow([f"ROI: {roi_name}"])
                    writer.writerow(["Time", "Volume", "Flow"])
                    for t, v, f in zip(result.time, result.volume, result.flow):
                        writer.writerow([t, v, f])
                    writer.writerow([])  # Blank line between ROIs
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save CSV: {e}")

    def _save_plot(self, ax, path):
        fig = ax.figure
        fig.savefig(path)
