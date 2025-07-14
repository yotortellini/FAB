# controllers/lighting_correction_controller.py

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Tuple

from PIL import Image, ImageTk
import numpy as np

from engine.video_engine import VideoEngine
from models.video_session import VideoSession
from helpers.lighting_correction import sample_background_profile, apply_lighting_correction

class LightingCorrectionController:
    """
    Step 6: Lighting Correction
    - Uses a fixed 800×450 canvas.
    - Draw a blank region (blue box), then shows side-by-side preview.
    - Both Apply and Skip simply call on_complete to progress.
    """
    PREVIEW_SIZE = (800, 450)

    def __init__(
        self,
        parent: tk.Frame,
        session: VideoSession,
        engine: VideoEngine,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        self.parent = parent
        self.session = session
        self.engine = engine
        self.on_complete = on_complete or (lambda: None)

        self.canvas = tk.Canvas(
            parent, width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            cursor="cross", bg="black"
        )
        self.canvas.pack()

        btns = ttk.Frame(parent)
        btns.pack(pady=5)
        # both buttons now simply advance the wizard
        ttk.Button(btns, text="Apply Correction", command=self._on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Skip",             command=self._on_skip).pack(side=tk.LEFT, padx=5)

        # drawing state
        self.start_x = self.start_y = 0
        self.curr_rect_id = None
        self.rect: Optional[Tuple[int,int,int,int]] = None

        self._tk_image = None
        self._load_frame()

        self.canvas.bind("<ButtonPress-1>",    self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",        self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouse_up)

    def _load_frame(self):
        frame = self.engine.get_frame(self.session.start_frame)
        if self.session.rotation_matrix is not None:
            frame = self.engine.apply_rotation(frame, self.session.rotation_matrix)

        h0, w0 = frame.shape[:2]
        pw, ph = self.PREVIEW_SIZE
        scale = min(pw/w0, ph/h0)
        nw, nh = int(w0*scale), int(h0*scale)
        xoff, yoff = (pw-nw)//2, (ph-nh)//2

        pil = Image.fromarray(frame[:,:,::-1])
        disp = pil.resize((nw, nh), Image.LANCZOS)
        canvas_img = Image.new("RGB", self.PREVIEW_SIZE, (0,0,0))
        canvas_img.paste(disp, (xoff, yoff))

        self._base = canvas_img
        self._xoff, self._yoff, self._scale = xoff, yoff, scale

        self._tk_image = ImageTk.PhotoImage(canvas_img)
        self.canvas.create_image(0,0,anchor="nw",image=self._tk_image)

    def _on_mouse_down(self, e):
        self.start_x, self.start_y = e.x, e.y
        if self.curr_rect_id:
            self.canvas.delete(self.curr_rect_id)
            self.curr_rect_id = None

    def _on_mouse_drag(self, e):
        if self.curr_rect_id:
            self.canvas.delete(self.curr_rect_id)
        self.curr_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, e.x, e.y,
            outline="blue", width=2
        )

    def _on_mouse_up(self, e):
        if not self.curr_rect_id:
            return
        x0, y0, x1, y1 = self.canvas.coords(self.curr_rect_id)
        ox = int((x0 - self._xoff)/self._scale)
        oy = int((y0 - self._yoff)/self._scale)
        ow = int((x1 - x0)/self._scale)
        oh = int((y1 - y0)/self._scale)
        self.rect = (ox, oy, ow, oh)
        self._preview_correction()

    def _preview_correction(self):
        if not self.rect:
            return
        profile = sample_background_profile(self.session, self.engine, self.rect)
        mid = (self.session.start_frame + self.session.end_frame)//2
        orig = self.engine.get_frame(mid)
        if self.session.rotation_matrix is not None:
            orig = self.engine.apply_rotation(orig, self.session.rotation_matrix)
        corr = apply_lighting_correction(orig, profile, mid)

        combined = np.concatenate([orig, corr], axis=1)
        h2, w2 = combined.shape[:2]
        pw, ph = self.PREVIEW_SIZE
        scale = min(pw/w2, ph/h2)
        nw, nh = int(w2*scale), int(h2*scale)
        pil = Image.fromarray(combined[:,:,::-1])
        disp = pil.resize((nw, nh), Image.LANCZOS)
        canvas_img = Image.new("RGB", self.PREVIEW_SIZE, (0,0,0))
        xoff, yoff = (pw-nw)//2, (ph-nh)//2
        canvas_img.paste(disp, (xoff, yoff))

        self._tk_image = ImageTk.PhotoImage(canvas_img)
        self.canvas.create_image(0,0,anchor="nw",image=self._tk_image)

    def _on_apply(self):
        if self.rect is None:
            messagebox.showerror("Error", "Please draw a blank region first.")
            return
        profile = sample_background_profile(self.session, self.engine, self.rect)
        self.session.background_profile = profile
        self.session.background_rect = self.rect
        messagebox.showinfo("Lighting Correction", "Background profile stored.")
        self.on_complete()

    def _on_skip(self):
        # no crash, just advance
        self.on_complete()
