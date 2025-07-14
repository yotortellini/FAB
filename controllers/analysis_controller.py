# controllers/analysis_controller.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional
import numpy as np
import os, csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFont

from models.video_session import VideoSession
from models.analysis_results import AnalysisResults
from engine.video_engine import VideoEngine
from controllers.lighting_correction_controller import LightingCorrectionController

class AnalysisController:
    """
    Step 6: Analysis
    - Runs analysis for *all* ROIs.
    - Plots volume curves together with legend.
    - Shows a status label updating which ROI is being processed.
    """
    def __init__(self,
                 parent: tk.Frame,
                 session: VideoSession,
                 engine: VideoEngine,
                 on_complete: Optional[Callable[[], None]] = None,
                 on_edit_rois: Optional[Callable[[], None]] = None):
        self.parent       = parent
        self.session      = session
        self.engine       = engine
        self.on_complete  = on_complete or (lambda: None)
        self.on_edit_rois = on_edit_rois or (lambda: None)

        self.results = {}
        self._build_ui()

    def _build_ui(self):
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

        # Confirm
        self.confirm_btn = ttk.Button(self.parent, text="Confirm Analysis",
                                      state="disabled", command=self._on_confirm)
        self.confirm_btn.pack(pady=10)

    def _mark_dirty(self):
        self.run_btn.config(state="normal")
        self.confirm_btn.config(state="disabled")
        self.save_btn.config(state="disabled")

    def _open_lighting(self):
        LightingCorrectionController(
            parent=self.parent,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: messagebox.showinfo("Lighting","Correction applied.")
        )

    def _on_run(self):
        self.run_btn.config(state="disabled")
        rois = self.session.rois
        if not rois:
            messagebox.showerror("Error","No ROIs defined.")
            self.run_btn.config(state="normal")
            return

        iv = self.interval_var.get()
        frame_range = range(self.session.start_frame, self.session.end_frame, iv)
        steps = len(frame_range)
        self.prog.config(maximum=steps, value=0)

        # Clear prior
        self.results.clear()
        self.ax_vol.clear()
        self.ax_flow.clear()
        self.status_label.config(text="")

        # base time array
        base_times = np.arange(steps) * iv / self.session.fps
        times = base_times * getattr(self.session, 'time_multiplier', 1.0)

        # loop over ROIs
        for name, roi in rois.items():
            # update status
            self.status_label.config(text=f"Analyzing: {name}")
            samples = []
            for idx in frame_range:
                frame = self.engine.get_frame(idx)
                if getattr(self.session, 'rotation_matrix', None) is not None:
                    frame = self.engine.apply_rotation(frame, self.session.rotation_matrix)
                x,y,w,h = roi.rect
                patch = frame[y:y+h, x:x+w]
                ch = {'B':0,'G':1,'R':2}[roi.channel]
                samples.append(patch[:,:,ch].mean())
                self.prog.step(1)
                self.parent.update_idletasks()

            arr = np.array(samples)
            norm   = (arr - arr.min()) / (arr.ptp() or 1)
            volume = norm * roi.total_volume
            flow   = np.gradient(volume, times)

            self.results[name] = (times, volume, flow)

            # incremental plotting
            w = max(1, self.smooth_var.get())
            vol_s  = np.convolve(volume, np.ones(w)/w, mode='same')
            flow_s = np.convolve(flow,   np.ones(w)/w, mode='same')
            self.ax_vol.plot(times, vol_s, label=name)
            self.ax_flow.plot(times, flow_s, label=name)
            self.canvas_vol.draw()
            self.canvas_flow.draw()

            # reset progress bar for next ROI
            self.prog.config(value=0)

        # finalize axes labels, ticks & legend
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
        self.status_label.config(text="")

        # enable confirm & save
        self.confirm_btn.config(state="normal")
        self.save_btn.config(state="normal")

    def _on_confirm(self):
        # store into session
        self.session.analysis_results = {
            name: AnalysisResults(time=t, volume=v, flow=f, smoothing_window=self.smooth_var.get())
            for name, (t,v,f) in self.results.items()
        }
        self.on_complete()
        messagebox.showinfo("Analysis","Analysis complete.")

    def _on_save(self):
        folder = filedialog.askdirectory(title="Save results to…")
        if not folder:
            return

        # CSV with one column per ROI
        csv_path = os.path.join(folder, "analysis_data.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["time"] + list(self.results.keys())
            writer.writerow(header)
            times = self.results[next(iter(self.results))][0]
            for i, t in enumerate(times):
                row = [t] + [self.results[n][1][i] for n in self.results]
                writer.writerow(row)

        # save plots
        self.ax_vol.figure.savefig(os.path.join(folder,"volume_plot.png"))
        self.ax_flow.figure.savefig(os.path.join(folder,"flow_plot.png"))

        # save ROI overlay
        mid = (self.session.start_frame + self.session.end_frame)//2
        frame = self.engine.get_frame(mid)
        if getattr(self.session, 'rotation_matrix', None) is not None:
            frame = self.engine.apply_rotation(frame, self.session.rotation_matrix)
        img = Image.fromarray(frame[:,:,::-1])
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for roi in self.session.rois.values():
            x,y,w,h = roi.rect
            draw.rectangle([x,y,x+w,y+h], outline="yellow", width=3)
            draw.text((x+4,y+4), roi.name, fill="yellow", font=font)
        img.save(os.path.join(folder,"roi_overlay.png"))

        messagebox.showinfo("Saved", f"Results saved to {folder}")
