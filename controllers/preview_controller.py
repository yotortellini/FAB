# controllers/preview_controller.py

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from models.video_session import VideoSession
from engine.video_engine import VideoEngine

class PreviewController:
    """
    Step 3: Framing Alignment
    - Overlays first and last frames at 50% opacity.
    - Letterboxes into a fixed 800×450 box.
    - Shows instructions and Continue/Exit buttons.
    """
    PREVIEW_SIZE = (800, 450)

    def __init__(self, parent: tk.Frame, session: VideoSession,
                 engine: VideoEngine, on_complete: callable):
        self.parent = parent
        self.session = session
        self.engine = engine
        self.on_complete = on_complete

        # grab first & last frames
        f0 = engine.get_frame(session.start_frame)
        f1 = engine.get_frame(session.end_frame - 1)

        # BGR→RGB → PIL RGBA
        pil0 = Image.fromarray(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)).convert("RGBA")
        pil1 = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # blend 50/50
        overlay = Image.blend(pil0, pil1, alpha=0.5)

        # scale & letterbox
        pw, ph = self.PREVIEW_SIZE
        w0, h0 = overlay.size
        scale = min(pw / w0, ph / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        disp = overlay.resize((new_w, new_h), Image.LANCZOS)
        canvas_img = Image.new("RGBA", self.PREVIEW_SIZE, (0, 0, 0, 255))
        x0 = (pw - new_w) // 2
        y0 = (ph - new_h) // 2
        canvas_img.paste(disp, (x0, y0), disp)
        final = canvas_img.convert("RGB")

        self._photo_overlay = ImageTk.PhotoImage(final)

        self.frame = tk.Frame(self.parent)
        self._build_ui()

    def _build_ui(self):
        ttk.Label(
            self.frame,
            text="The start and end frames you selected have been overlaid to highlight any shifts in the framing of the footage.",
            wraplength=700,
            justify="center"
        ).pack(pady=10)

        ttk.Label(self.frame, text="Framing Alignment", font=(None,16)).pack(pady=5)

        pw, ph = self.PREVIEW_SIZE
        self.canvas = tk.Canvas(self.frame, width=pw, height=ph, bg='black')
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo_overlay)

        ctrl = ttk.Frame(self.frame)
        ctrl.pack(pady=10)
        ttk.Button(ctrl, text="Continue", command=self._on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Exit",     command=self._on_exit).pack(side=tk.LEFT, padx=5)

        self.frame.pack(fill=tk.BOTH, expand=True)

    def _on_confirm(self):
        self.on_complete()

    def _on_exit(self):
        self.frame.winfo_toplevel().destroy()
