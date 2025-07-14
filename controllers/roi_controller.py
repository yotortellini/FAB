# controllers/roi_controller.py

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from typing import Callable, Optional

from models.video_session import VideoSession
from models.roi import ROI
from engine.video_engine import VideoEngine

class ROIController:
    PREVIEW_SIZE = (800, 450)

    def __init__(
        self,
        parent: tk.Frame,
        session: VideoSession,
        engine: VideoEngine,
        on_complete: Callable[[], None]
    ):
        self.parent = parent
        self.session = session
        self.engine = engine
        self.on_complete = on_complete

        if not hasattr(self.session, 'rois'):
            self.session.rois = {}

        # Prepare deskewed frame once
        frame = self.engine.get_frame(self.session.start_frame)
        if self.session.rotation_matrix is not None:
            frame = engine.apply_rotation(frame, self.session.rotation_matrix)
        h0, w0 = frame.shape[:2]
        pw, ph = self.PREVIEW_SIZE
        self._scale = min(pw/w0, ph/h0)
        self._xoff  = int((pw - w0*self._scale)//2)
        self._yoff  = int((ph - h0*self._scale)//2)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        disp = pil.resize((int(w0*self._scale), int(h0*self._scale)), Image.LANCZOS)
        canvas_im = Image.new("RGB", self.PREVIEW_SIZE, (0,0,0))
        canvas_im.paste(disp, (self._xoff, self._yoff))
        self._photo_disp = ImageTk.PhotoImage(canvas_im)

        # State
        self._rect_id = None
        self._overlay_ids = []
        self.selected_name: Optional[str] = None

        # Build UI
        self.frame = tk.Frame(self.parent)
        self._build_ui()
        self._refresh_list()
        self._draw_all_rois()

    def _build_ui(self):
        pw, ph = self.PREVIEW_SIZE
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instruction
        ttk.Label(main,
                  text="Draw a rectangle on the image to create a new ROI",
                  font=(None, 12, 'italic')).pack(pady=(0,10))

        # Left: list + delete
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="ROIs").pack()
        self.listbox = tk.Listbox(left, height=12)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        ttk.Button(left, text="Delete ROI", command=self._on_delete).pack(pady=5)

        # Right: canvas + form + save
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Video canvas
        self.canvas = tk.Canvas(right, width=pw, height=ph, bg='black')
        self.canvas.pack()
        self.canvas_img = self.canvas.create_image(0,0,anchor='nw',image=self._photo_disp)
        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Form for editing parameters
        form = ttk.Frame(right)
        form.pack(fill=tk.X, pady=5)
        save_frame = ttk.Frame(right)
        save_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(save_frame, text="Save ROI", command=self._save_edits).pack(side=tk.RIGHT)

        # Name (read‐only; renaming via popup)
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="e")
        self.name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.name_var, state="readonly")\
            .grid(row=0, column=1, sticky="w")

        # Channel
        ttk.Label(form, text="Channel").grid(row=1, column=0, sticky="e", pady=(5,0))
        self.channel_var = tk.StringVar(value='R')
        ttk.OptionMenu(form, self.channel_var, 'R','R','G','B')\
            .grid(row=1, column=1, sticky="w", pady=(5,0))

        # Total Volume
        ttk.Label(form, text="Total Volume (µL)").grid(row=2, column=0, sticky="e", pady=(5,0))
        self.vol_var = tk.DoubleVar(value=0.0)
        ttk.Entry(form, textvariable=self.vol_var, width=8)\
            .grid(row=2, column=1, sticky="w", pady=(5,0))

        # Start frac
        ttk.Label(form, text="Start frac").grid(row=3, column=0, sticky="e", pady=(5,0))
        self.start_frac = tk.DoubleVar(value=0.0)
        ttk.Spinbox(form, from_=0.0, to=1.0, increment=0.01,
                    textvariable=self.start_frac, width=6)\
            .grid(row=3, column=1, sticky="w", pady=(5,0))

        # End frac
        ttk.Label(form, text="End frac").grid(row=4, column=0, sticky="e", pady=(5,0))
        self.end_frac = tk.DoubleVar(value=1.0)
        ttk.Spinbox(form, from_=0.0, to=1.0, increment=0.01,
                    textvariable=self.end_frac, width=6)\
            .grid(row=4, column=1, sticky="w", pady=(5,0))

        self.frame.pack(fill=tk.BOTH, expand=True)

    def _on_canvas_press(self, e):
        self._press = (e.x, e.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_canvas_drag(self, e):
        x0, y0 = self._press
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, e.x, e.y, outline='red', width=2
        )

    def _on_canvas_release(self, e):
        if not self._rect_id:
            return

        # Name prompt
        name = simpledialog.askstring("New ROI", "Enter a name for this ROI:")
        if not name:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
            return

        # Map rectangle to original coords
        x0, y0, x1, y1 = self.canvas.coords(self._rect_id)
        ox = int((x0 - self._xoff)/self._scale)
        oy = int((y0 - self._yoff)/self._scale)
        ow = int((x1 - x0)/self._scale)
        oh = int((y1 - y0)/self._scale)

        # Create ROI with defaults
        roi = ROI(
            name=name,
            rect=(ox, oy, ow, oh),
            channel='R',
            total_volume=0.0,
            start_frac=0.0,
            end_frac=1.0,
            interval=1,
            smoothing_window=1
        )
        self.session.rois[name] = roi
        self.selected_name = name

        # Refresh display & select it
        self._refresh_list()
        self._draw_all_rois()
        self._load_selected(name)

        # Unlock Next on first ROI
        if len(self.session.rois) == 1:
            self.on_complete()

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for nm in self.session.rois:
            self.listbox.insert(tk.END, nm)

    def _draw_all_rois(self):
        for oid in self._overlay_ids:
            self.canvas.delete(oid)
        self._overlay_ids.clear()
        for roi in self.session.rois.values():
            x,y,w,h = roi.rect
            cx,cy = x*self._scale + self._xoff, y*self._scale + self._yoff
            cw,ch = w*self._scale, h*self._scale
            oid = self.canvas.create_rectangle(
                cx, cy, cx+cw, cy+ch,
                outline='green', width=2
            )
            self._overlay_ids.append(oid)

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        self._load_selected(name)

    def _load_selected(self, name: str):
        roi = self.session.rois[name]
        self.selected_name = name

        # Populate form
        self.name_var.set(roi.name)
        self.channel_var.set(roi.channel)
        self.vol_var.set(roi.total_volume)
        self.start_frac.set(roi.start_frac)
        self.end_frac.set(roi.end_frac)

        # Highlight its rectangle
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        x,y,w,h = roi.rect
        cx,cy = x*self._scale + self._xoff, y*self._scale + self._yoff
        cw,ch = w*self._scale, h*self._scale
        self._rect_id = self.canvas.create_rectangle(
            cx, cy, cx+cw, cy+ch,
            outline='red', width=2
        )

    def _save_edits(self):
        """Save changes to the currently selected ROI."""
        if not self.selected_name:
            messagebox.showerror("Error", "Select an ROI first before saving.")
            return

        roi = self.session.rois[self.selected_name]
        # Update fields
        roi.channel      = self.channel_var.get()
        roi.total_volume = self.vol_var.get()
        roi.start_frac   = float(self.start_frac.get())
        roi.end_frac     = float(self.end_frac.get())

        # Redraw overlays
        self._refresh_list()
        self._draw_all_rois()

    def _on_delete(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        del self.session.rois[name]
        self.selected_name = None
        # clear form & rectangle
        self.name_var.set("")
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self._refresh_list()
        self._draw_all_rois()
