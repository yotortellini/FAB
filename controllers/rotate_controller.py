# controllers/rotate_controller.py

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class RotateController:
    """
    Step 4: Rotate / Deskew
    - Click+drag across a horizontal feature to measure its angle.
    - Deskews by rotating by the negative of that measured angle,
      so your red guideline stays aligned.
    - The red line + its end-point dots rotate together.
    - Reset or Confirm to lock in the rotation_matrix.
    """
    PREVIEW_SIZE = (800, 450)

    def __init__(self,
                 parent: tk.Frame,
                 session: VideoSession,
                 engine: VideoEngine,
                 on_complete: callable):
        self.parent = parent
        self.session = session
        self.engine = engine
        self.on_complete = on_complete

        # Load mid-clip frame and apply any existing rotation
        mid = (session.start_frame + session.end_frame) // 2
        frame = engine.get_frame(mid)
        if session.rotation_matrix is not None:
            frame = engine.apply_rotation(frame, session.rotation_matrix)
        self.orig_img = frame
        self.h0, self.w0 = frame.shape[:2]

        # Compute scale & offset to fit PREVIEW_SIZE
        pw, ph = self.PREVIEW_SIZE
        scale = min(pw / self.w0, ph / self.h0)
        self._scale = scale
        self._xoff = int((pw - self.w0 * scale) // 2)
        self._yoff = int((ph - self.h0 * scale) // 2)

        # Prepare initial display image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        disp = pil.resize((int(self.w0 * scale), int(self.h0 * scale)), Image.LANCZOS)
        canvas_img = Image.new("RGB", self.PREVIEW_SIZE, (0, 0, 0))
        canvas_img.paste(disp, (self._xoff, self._yoff))
        self._photo_disp = ImageTk.PhotoImage(canvas_img)

        # Drawing state
        self._start = None
        self._line = None
        self._dot1 = None
        self._dot2 = None
        self._orig0 = None
        self._orig1 = None

        self.frame = tk.Frame(self.parent)
        self._build_ui()

    def _build_ui(self):
        ttk.Label(self.frame,
            text="Click and drag to draw a line across a horizontal feature\n"
                 "(e.g. chip edge) to deskew.",
            wraplength=700, justify="center"
        ).pack(pady=10)
        ttk.Label(self.frame, text="Rotate / Deskew", font=(None,16)).pack(pady=5)

        pw, ph = self.PREVIEW_SIZE
        self.canvas = tk.Canvas(self.frame, width=pw, height=ph, cursor="cross")
        self.canvas.pack()
        self.canvas_img = self.canvas.create_image(0, 0, anchor='nw', image=self._photo_disp)

        self.canvas.bind("<ButtonPress-1>",    self._on_press)
        self.canvas.bind("<B1-Motion>",        self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_release)

        ctrl = ttk.Frame(self.frame)
        ctrl.pack(pady=5, fill=tk.X, padx=10)
        ttk.Label(ctrl, text="Angle to apply (°):").pack(side=tk.LEFT)
        self.angle_var = tk.DoubleVar(value=0.0)
        self.angle_spin = ttk.Spinbox(
            ctrl, from_=-180, to=180, increment=0.1,
            textvariable=self.angle_var,
            command=self._on_angle_change,
            state="disabled", width=7
        )
        self.angle_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=5)
        self.confirm_btn = ttk.Button(
            ctrl, text="Confirm Rotation",
            command=self._on_confirm,
            state="disabled"
        )
        self.confirm_btn.pack(side=tk.RIGHT)

        self.frame.pack(fill=tk.BOTH, expand=True)

    def _on_press(self, e):
        # start of drag
        self._start = (e.x, e.y)
        # clear previous decorations
        for attr in ("_line", "_dot1", "_dot2"):
            obj = getattr(self, attr)
            if obj:
                self.canvas.delete(obj)
                setattr(self, attr, None)
        self.confirm_btn.config(state="disabled")
        self.angle_spin.config(state="disabled")

    def _on_drag(self, e):
        # rubber‐band line
        if self._line:
            self.canvas.delete(self._line)
        x0, y0 = self._start
        self._line = self.canvas.create_line(x0, y0, e.x, e.y,
                                             fill="red", width=2)

    def _on_release(self, e):
        # drop end‐point dots
        x0, y0 = self._start
        x1, y1 = e.x, e.y
        r = 4
        self._dot1 = self.canvas.create_oval(x0-r, y0-r, x0+r, y0+r,
                                             fill="red", outline="red")
        self._dot2 = self.canvas.create_oval(x1-r, y1-r, x1+r, y1+r,
                                             fill="red", outline="red")

        # map canvas coords back to image coords
        ox0 = (x0 - self._xoff) / self._scale
        oy0 = (y0 - self._yoff) / self._scale
        ox1 = (x1 - self._xoff) / self._scale
        oy1 = (y1 - self._yoff) / self._scale
        self._orig0 = (ox0, oy0)
        self._orig1 = (ox1, oy1)

        # measured slope angle (positive CCW)
        raw = np.degrees(np.arctan2(oy1 - oy0, ox1 - ox0))
        # deskew by rotating by the negative of raw
        angle_to_apply = raw
        self.angle_var.set(round(angle_to_apply, 2))
        self.angle_spin.config(state="normal")
        self._apply_rotation(angle_to_apply)
        self.confirm_btn.config(state="normal")

    def _apply_rotation(self, angle: float):
        # build and store rotation matrix
        center = (self.w0/2, self.h0/2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.session.rotation_matrix = rot_mat

        # rotate the original image
        rotated = cv2.warpAffine(self.orig_img, rot_mat,
                                 (self.w0, self.h0),
                                 flags=cv2.INTER_LINEAR)

        # update display image
        pil = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        disp = pil.resize((int(self.w0*self._scale), int(self.h0*self._scale)), Image.LANCZOS)
        canvas_img = Image.new("RGB", self.PREVIEW_SIZE, (0,0,0))
        canvas_img.paste(disp, (self._xoff, self._yoff))
        self._photo_rot = ImageTk.PhotoImage(canvas_img)
        self.canvas.itemconfig(self.canvas_img, image=self._photo_rot)

        # rotate the red guideline and dots along with the image
        if self._orig0 and self._orig1:
            a, b, c = rot_mat[0]
            d, e, f = rot_mat[1]
            nx0 = a*self._orig0[0] + b*self._orig0[1] + c
            ny0 = d*self._orig0[0] + e*self._orig0[1] + f
            nx1 = a*self._orig1[0] + b*self._orig1[1] + c
            ny1 = d*self._orig1[0] + e*self._orig1[1] + f
            cx0 = nx0*self._scale + self._xoff
            cy0 = ny0*self._scale + self._yoff
            cx1 = nx1*self._scale + self._xoff
            cy1 = ny1*self._scale + self._yoff

            # clear old
            for attr in ("_line", "_dot1", "_dot2"):
                obj = getattr(self, attr)
                if obj:
                    self.canvas.delete(obj)
                    setattr(self, attr, None)

            # draw new line and dots
            self._line = self.canvas.create_line(cx0, cy0, cx1, cy1,
                                                 fill="red", width=2)
            r = 4
            self._dot1 = self.canvas.create_oval(cx0-r, cy0-r, cx0+r, cy0+r,
                                                 fill="red", outline="red")
            self._dot2 = self.canvas.create_oval(cx1-r, cy1-r, cx1+r, cy1+r,
                                                 fill="red", outline="red")

    def _on_angle_change(self):
        if self.angle_spin["state"] == "normal":
            self._apply_rotation(self.angle_var.get())

    def _reset(self):
        for attr in ("_line", "_dot1", "_dot2"):
            obj = getattr(self, attr)
            if obj:
                self.canvas.delete(obj)
                setattr(self, attr, None)
        # restore original display
        self.canvas.itemconfig(self.canvas_img, image=self._photo_disp)
        self.angle_var.set(0.0)
        self.angle_spin.config(state="disabled")
        self.confirm_btn.config(state="disabled")
        self.session.rotation_matrix = None
        self._orig0 = None
        self._orig1 = None

    def _on_confirm(self):
        self.on_complete()
