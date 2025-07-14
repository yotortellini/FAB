# controllers/ui_controller.py

# controllers/ui_controller.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict

from models.video_session import VideoSession
from engine.video_engine import VideoEngine
from controllers.scan_controller import ScanController
from controllers.preview_controller import PreviewController
from controllers.rotate_controller import RotateController
from controllers.roi_controller import ROIController
from controllers.analysis_controller import AnalysisController

class UIController:
    """
    Wizard steps:
      1. Load
      2. Scan
      3. Preview
      4. Rotate
      5. ROIs
      6. Analysis
    """
    def __init__(self, master: tk.Tk, session: VideoSession, engine: VideoEngine):
        self.master = master
        self.session = session
        self.engine = engine

        self.master.geometry("1024x768")
        self.master.resizable(False, False)



        # ——— HEADER WITH LOGO —————————————————————————————————————
        from pathlib import Path
        from PIL import Image, ImageTk

        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        img = Image.open(logo_path)

        # Resize to a small corner logo:
        img.thumbnail((30, 30), Image.LANCZOS)

        self._logo_tk = ImageTk.PhotoImage(img)

        logo_lbl = tk.Label(self.master, image=self._logo_tk, bd=0)
        # relx=1.0 means 100% to the right, y=0 at the top, anchor='ne' pins its north-east corner
        logo_lbl.place(relx=0.8, y=25, anchor='ne')
        # ——————————————————————————————————————————————————————————

        self.steps = {}            # type: Dict[int, tk.Frame]
        self.current_step = 1
        self.total_steps  = 6

        self.container = tk.Frame(self.master)
        self.container.pack(fill=tk.BOTH, expand=True)

        nav = tk.Frame(self.master)
        nav.pack(fill=tk.X, side=tk.BOTTOM)
        self.back_button = tk.Button(nav, text="Back",   command=self.go_back, state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT,  padx=5, pady=5)
        self.next_button = tk.Button(nav, text="Next",   command=self.go_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.build_steps()
        self.show_step(1)
        
        logo_lbl = tk.Label(self.master, image=self._logo_tk, bd=0)
        logo_lbl.place(relx=1.0, rely=0.0, anchor='ne')
        logo_lbl.lift()  # bring to front

    def build_steps(self):
        self.steps = {1: self.build_load_video_step(self.container)}

    def show_step(self, step:int):
        for f in self.steps.values(): f.pack_forget()
        frm = self.steps.get(step)
        if frm: frm.pack(fill=tk.BOTH, expand=True)
        # if we're showing Analysis (step 6) and it's already built, mark it dirty
        if step == 6 and hasattr(self, 'analysis_ctrl'):
            # assumes AnalysisController has a public 'mark_dirty' method
            self.analysis_ctrl._mark_dirty()

        self.back_button['state'] = tk.NORMAL if step>1 else tk.DISABLED
        self.next_button['state'] = (
            tk.NORMAL if step<self.total_steps and self.is_step_complete(step)
            else tk.DISABLED
        )
        self.current_step = step

    def go_next(self):
        ns = self.current_step+1
        if ns not in self.steps:
            if   ns==2: self.steps[2] = self.build_temporal_scan_step(self.container)
            elif ns==3: self.steps[3] = self.build_preview_video_step(self.container)
            elif ns==4: self.steps[4] = self.build_rotate_step(self.container)
            elif ns==5: self.steps[5] = self.build_define_rois_step(self.container)
            elif ns==6: self.steps[6] = self.build_analysis_step(self.container)
        self.show_step(ns)

    def go_back(self):
        if self.current_step>1:
            self.show_step(self.current_step-1)

    def is_step_complete(self, step:int)->bool:
        if step==1: return bool(getattr(self.session,'path',None))
        if step in (2,3,4): return self.next_button['state']==tk.NORMAL
        if step==5: return bool(getattr(self.session,'rois',{}))
        if step==6: return getattr(self.session,'analysis_results',None) is not None
        return False

    # -------------------------------------------------------------------------
    # Step 1: Load Video
    # -------------------------------------------------------------------------
    def build_load_video_step(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent)
        tk.Label(frame, text="Step 1: Load Video", font=(None, 16)).pack(pady=10)

        file_frame = tk.Frame(frame)
        file_frame.pack(pady=5)
        self.path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.path_var, width=50, state='readonly').pack(side=tk.LEFT)
        tk.Button(file_frame, text="Browse…", command=self.on_browse).pack(side=tk.LEFT, padx=5)

        tm_frame = tk.Frame(frame)
        tm_frame.pack(pady=5)
        tk.Label(tm_frame, text="Time Multiplier:").pack(side=tk.LEFT)
        self.tm_var = tk.DoubleVar(value=1.0)
        sb = tk.Spinbox(tm_frame, from_=0.01, to=100.0, increment=0.01,
                        textvariable=self.tm_var, width=6)
        sb.pack(side=tk.LEFT)

        # keep session in sync if user edits multiplier later
        def _on_tm_change(*_):
            try:
                self.session.time_multiplier = float(self.tm_var.get())
            except (ValueError, tk.TclError):
        # ignore if user is mid-edit or the field is empty
                pass
        self.tm_var.trace_add('write', _on_tm_change)

        meta_frame = tk.Frame(frame)
        meta_frame.pack(pady=10)
        self.fps_var = tk.StringVar(value="FPS: N/A")
        self.count_var = tk.StringVar(value="Frames: N/A")
        tk.Label(meta_frame, textvariable=self.fps_var).pack()
        tk.Label(meta_frame, textvariable=self.count_var).pack()

        return frame

    def on_browse(self):
        
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return
        try:
            meta = self.engine.load_video(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            return

        # Populate session
        self.session.path = path
        self.session.fps = meta['fps']
        self.session.frame_count = int(meta['frame_count'])
        self.session.start_frame = 0
        self.session.end_frame = self.session.frame_count
        self.session.time_multiplier = float(self.tm_var.get())        

        print(f"[DEBUG] time_multiplier set to {self.session.time_multiplier}")

        self.video_loaded = True
        


        # Update UI
        self.path_var.set(path)
        self.fps_var.set(f"FPS: {self.session.fps:.2f}")
        self.count_var.set(f"Frames: {self.session.frame_count}")
        self.next_button.config(state=tk.NORMAL)

    # -------------------------------------------------------------------------
    # Step 2: Temporal Scan
    # -------------------------------------------------------------------------
    def build_temporal_scan_step(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent)
        self.next_button.config(state=tk.DISABLED)
        self.scan_ctrl = ScanController(
            parent=frame,
            session=self.session,
            engine=self.engine,
            on_run=lambda: self.next_button.config(state=tk.DISABLED),
            on_complete=lambda: self.next_button.config(state=tk.NORMAL),
            max_samples=200,
            sample_interval=None
        )
        return frame

    # -------------------------------------------------------------------------
    # Step 3: Framing Alignment
    # -------------------------------------------------------------------------
    def build_preview_video_step(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent)
        self.next_button.config(state=tk.DISABLED)
        self.preview_ctrl = PreviewController(
            parent=frame,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: self.next_button.config(state=tk.NORMAL)
        )
        return frame

    # -------------------------------------------------------------------------
    # Step 4: Rotate / Deskew
    # -------------------------------------------------------------------------
    def build_rotate_step(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent)
        self.next_button.config(state=tk.DISABLED)
        self.rotate_ctrl = RotateController(
            parent=frame,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: self.next_button.config(state=tk.NORMAL)
        )
        return frame

    # -------------------------------------------------------------------------
    # Step 5: Define ROIs
    # -------------------------------------------------------------------------
    def build_define_rois_step(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent)
        self.next_button.config(state=tk.DISABLED)
        self.roi_ctrl = ROIController(
            parent=frame,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: self.next_button.config(state=tk.NORMAL)
        )
        return frame

    # -------------------------------------------------------------------------
    # Step 6: Analysis UI (includes optional Lighting button).
    # -------------------------------------------------------------------------
 
    def build_analysis_step(self, parent):
        frame = tk.Frame(parent)
        self.next_button.config(state=tk.DISABLED)
        self.analysis_ctrl = AnalysisController(
            parent=frame,
            session=self.session,
            engine=self.engine,
            on_complete=lambda: self.next_button.config(state=tk.NORMAL),
            on_edit_rois=self.go_back
        )
        return frame