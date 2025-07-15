# controllers/scan_controller.py

import threading
import functools
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class ScanController:
    """
    Controller for Step 2: Temporal Scan.
    - "Run Scan" for background sampling with determinate progress.
    - Interactive plot with two red vertical lines & grey span.
    - Thumbnails outlined in black beside each slider (placeholder until ready).
    """

    THUMB_SIZE = (120, 90)
    PLACEHOLDER_COLOR = (220, 220, 220)

    def __init__(
        self,
        parent: tk.Frame,
        session: VideoSession,
        engine: VideoEngine,
        on_run: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        max_samples: int | None = None,
        sample_interval: int | None = None,
    ):
        self.parent = parent
        self.session = session
        self.engine = engine
        self.on_run = on_run or (lambda: None)
        self.on_complete = on_complete or (lambda: None)
        self.max_samples = max_samples
        self.sample_interval = sample_interval

        self.time = np.array([])
        self.intensity = np.array([])

        self.start_line = None
        self.end_line = None
        self.shade = None

        self.frame = tk.Frame(self.parent)
        self._build_ui()

    def _build_ui(self):
        ttk.Label(self.frame, text="Step 2: Temporal Scan", font=(None,16)).pack(pady=10)

        # Plot area
        fig = Figure(figsize=(6,3))
        self._ax = fig.add_subplot(111)
        self._ax.set_xlabel("Time")
        self._ax.set_ylabel("Light Intensity")
        self._ax.set_xticks([])    # remove numeric ticks
        self._ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Progress bar (determinate)
        self.progress = ttk.Progressbar(self.frame, mode='determinate', maximum=1)
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        self.progress.pack_forget()

        # Run Scan button
        self.run_btn = ttk.Button(self.frame, text="Run Scan", command=self._on_run)
        self.run_btn.pack(pady=5)

        # Start slider + thumbnail
        start_frame = ttk.Frame(self.frame)
        start_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(start_frame, text="Measurement Start").pack(side=tk.LEFT)
        self.start_slider = ttk.Scale(start_frame, from_=0.0, to=1.0, state=tk.DISABLED)
        self.start_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.start_thumb = ttk.Label(start_frame, borderwidth=1, relief="solid")
        self.start_thumb.pack(side=tk.LEFT, padx=10)

        # End slider + thumbnail
        end_frame = ttk.Frame(self.frame)
        end_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(end_frame, text="Measurement End").pack(side=tk.LEFT)
        self.end_slider = ttk.Scale(end_frame, from_=0.0, to=1.0, state=tk.DISABLED)
        self.end_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.end_thumb = ttk.Label(end_frame, borderwidth=1, relief="solid")
        self.end_thumb.pack(side=tk.LEFT, padx=10)

        # Placeholder images so UI doesn’t shift
        placeholder = Image.new("RGB", self.THUMB_SIZE, self.PLACEHOLDER_COLOR)
        self._placeholder_img = ImageTk.PhotoImage(placeholder)
        self.start_thumb.configure(image=self._placeholder_img)
        self.end_thumb.configure(image=self._placeholder_img)

        # Confirm Selection button
        self.confirm_btn = ttk.Button(self.frame, text="Confirm Selection", command=self._on_confirm)
        self.confirm_btn.pack(pady=10)
        self.confirm_btn.config(state=tk.DISABLED)

        self.frame.pack(fill=tk.BOTH, expand=True)

    def _on_run(self):
        """Start scanning in a background thread and keep UI responsive."""
        self.run_btn.config(state=tk.DISABLED)
        self.progress.config(value=0, maximum=1)
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        # Start the scan in a background thread
        threading.Thread(target=self._async_load, daemon=True).start()

    def _async_load(self):
        try:
            start, end = self.session.start_frame, self.session.end_frame
            total = end - start
            if total <= 0 or self.engine.cap is None:
                raise RuntimeError("No video loaded")

            if self.sample_interval and self.sample_interval > 1:
                indices = np.arange(start, end, self.sample_interval, dtype=int)
            elif self.max_samples and self.max_samples < total:
                indices = np.linspace(start, end-1, self.max_samples, dtype=int)
            else:
                indices = np.arange(start, end, dtype=int)

            count = len(indices)
            self.frame.after(0, lambda: self.progress.config(maximum=count))

            intensities = []
            for i, idx in enumerate(indices, start=1):
                frame = self.engine.get_frame(idx)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensities.append(float(gray.mean()))
                self.frame.after(0, functools.partial(self.progress.config, value=i))

            base_time = (indices - start) / self.session.fps
            self.time = base_time * getattr(self.session, 'time_multiplier', 1.0)
            self.intensity = np.array(intensities)

            self.frame.after(0, self._on_data_ready)
        except Exception as e:
            self.frame.after(0, lambda: messagebox.showerror("Scan Error", str(e)))
            self.frame.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _on_data_ready(self):
        self.progress.pack_forget()

        self.start_slider.config(state=tk.NORMAL, command=self._on_slider_move)
        self.end_slider.config(state=tk.NORMAL, command=self._on_slider_move)
        self.end_slider.set(1.0)

        self._ax.clear()
        self._ax.plot(self.time, self.intensity)
        self._ax.set_xlabel("Time")
        self._ax.set_ylabel("Light Intensity")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.start_line = self._ax.axvline(0, color='red')
        self.end_line   = self._ax.axvline(self.time[-1], color='red')
        self.shade      = self._ax.axvspan(0, self.time[-1], color='grey', alpha=0.3)
        self.canvas.draw()

        self.confirm_btn.config(state=tk.NORMAL)

    def _on_slider_move(self, _=None):
        if self.start_line is None or self.end_line is None:
            return
        frac_s = float(self.start_slider.get())
        frac_e = float(self.end_slider.get())
        t0 = frac_s * self.time[-1]
        t1 = frac_e * self.time[-1]

        self.start_line.set_xdata([t0, t0])
        self.end_line.set_xdata([t1, t1])
        self.shade.remove()
        self.shade = self._ax.axvspan(t0, t1, color='grey', alpha=0.3)
        self.canvas.draw()
        self._update_thumbnails()

    def _update_thumbnails(self):
        total = self.session.end_frame - self.session.start_frame
        frac_s = float(self.start_slider.get())
        frac_e = float(self.end_slider.get())
        idx_s = self.session.start_frame + int(frac_s*(total-1))
        idx_e = self.session.start_frame + int(frac_e*(total-1))

        frame_s = self.engine.get_frame(idx_s)
        img_s = Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)) \
                 .resize(self.THUMB_SIZE)
        self._tk_s = ImageTk.PhotoImage(img_s)
        self.start_thumb.configure(image=self._tk_s)

        frame_e = self.engine.get_frame(idx_e)
        img_e = Image.fromarray(cv2.cvtColor(frame_e, cv2.COLOR_BGR2RGB)) \
                 .resize(self.THUMB_SIZE)
        self._tk_e = ImageTk.PhotoImage(img_e)
        self.end_thumb.configure(image=self._tk_e)

    def _on_confirm(self):
        frac_s = float(self.start_slider.get())
        frac_e = float(self.end_slider.get())
        total = self.session.end_frame - self.session.start_frame
        new_start = self.session.start_frame + int(frac_s*(total-1))
        new_end   = self.session.start_frame + int(frac_e*(total-1))
        self.session.set_frame_range(new_start, new_end)
        self.on_complete()

    def _on_scan(self):
        """Start scanning in a background thread."""
        self.scan_btn.config(state="disabled")
        thread = threading.Thread(target=self._run_scan)
        thread.start()

    def _run_scan(self):
        self.run_btn.config(state=tk.DISABLED)
        self.progress.config(value=0, maximum=1)
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        self.on_run()
        threading.Thread(target=self._async_load, daemon=True).start()
