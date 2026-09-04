import os
import time
import json
import threading
import math
from datetime import datetime
from collections import Counter

import cv2
from ultralytics import YOLO

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


# ==============================
# CONFIG (EDIT THESE PATHS)
# ==============================

MODEL_PATH = r"C:\Users\HP\Desktop\AI_Project\6_RealTimeApp\best.pt"
CLASS_NAMES = ["bottle", "dry", "wet"]

CLASS_IMAGE_PATHS = {
    "bottle": r"C:\Users\HP\Desktop\AI_Project\6_RealTimeApp\bottle.png",
    "dry":    r"C:\Users\HP\Desktop\AI_Project\6_RealTimeApp\dry.png",
    "wet":    r"C:\Users\HP\Desktop\AI_Project\6_RealTimeApp\wet.jpg",
}

"""CAM_INDEX = 0 #1
CAM_BACKEND = cv2.CAP_MSMF"""

IP = "172.20.144.127"  # IP 
PORT = "4747"         # Port 



"""def find_available_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return 0
CAM_INDEX = find_available_camera()

"""



FEEDBACK_DIR = r"C:\Users\HP\Desktop\AI_Project\EXT"
# ==============================
# APP
# ==============================

class TrashDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trash Detection")
        self.root.geometry("1100x720")

        self.model = YOLO(MODEL_PATH)

        #self.cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
        self.cap = cv2.VideoCapture(f"http://{IP}:{PORT}/video")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.preview_photo = None
        self.pic_photo = None
        self.class_photo = None
        self.class_imgs = {}

        self._overlay_lock = threading.Lock()
        self.live_box = None
        self.live_label = ""
        self.live_conf = 0.0
        self.live_seen_ts = 0.0
        
        self.live_mode = False
        self._live_thread = None

        self.latest_frame_lock = threading.Lock()
        self.latest_frame_bgr = None

        self.last_sample = None
        
        self.live_freeze = False
        self.live_frozen_frame_bgr = None
        self.running_best_scan = False

        self._preload_class_images()
        self._build_ui()

        self._update_camera_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.show_menu()

    def _preload_class_images(self):
        for cls, path in CLASS_IMAGE_PATHS.items():
            try:
                img = Image.open(path).convert("RGBA")
                img = img.resize((220, 220), Image.LANCZOS)
                self.class_imgs[cls] = ImageTk.PhotoImage(img)
            except Exception:
                self.class_imgs[cls] = None

    def _build_ui(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        self.container = ttk.Frame(self.root, padding=12)
        self.container.pack(fill="both", expand=True)

        # ======================
        # MENU FRAME
        # ======================
        
        self.menu_frame = ttk.Frame(self.container)
        ttk.Label(self.menu_frame, text="Trash Detection", font=("Segoe UI", 20, "bold")).pack(pady=(10, 20))

        btns = ttk.Frame(self.menu_frame)
        btns.pack(pady=10)

        self.btn_live = ttk.Button(btns, text="Real-time Detection", command=self.show_live, width=26)
        self.btn_live.pack(pady=8)

        self.btn_pic = ttk.Button(btns, text="Picture Detection", command=self.show_picture, width=26)
        self.btn_pic.pack(pady=8)

        ttk.Label(
            self.menu_frame,
            text="Choose a mode.\nReal-time uses webcam. Picture detection loads an image file.",
            font=("Segoe UI", 10),
            justify="center"
        ).pack(pady=(20, 0))

        # ======================
        # LIVE FRAME
        # ======================
        
        self.live_frame = ttk.Frame(self.container)

        live_top = ttk.Frame(self.live_frame)
        live_top.pack(fill="x")

        ttk.Label(live_top, text="Real-time Detection", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.live_back_btn = ttk.Button(live_top, text="Return to Menu", command=self._leave_live_to_menu)
        self.live_back_btn.pack(side="right")

        live_body = ttk.Frame(self.live_frame)
        live_body.pack(fill="both", expand=True, pady=(10, 0))

        self.live_preview_label = ttk.Label(live_body, text="Starting camera...")
        self.live_preview_label.pack(side="left", fill="both", expand=True)

        live_right = ttk.Frame(live_body, width=340)
        live_right.pack(side="right", fill="y", padx=(12, 0))

        ttk.Label(live_right, text="Live status", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))
        self.live_status_var = tk.StringVar(value="Ready.")
        ttk.Label(live_right, textvariable=self.live_status_var, font=("Segoe UI", 11)).pack(pady=(0, 10))

        # Buttons: Best Scan + Resume
        scan_btns = ttk.Frame(live_right)
        scan_btns.pack(fill="x", pady=(4, 8))

        self.best_scan_btn = ttk.Button(scan_btns, text="Best Scan Result (3s)", command=self.start_best_scan_3s)
        self.best_scan_btn.pack(fill="x", pady=(0, 6))

        self.resume_btn = ttk.Button(scan_btns, text="Resume Live", command=self.resume_live, state="disabled")
        self.resume_btn.pack(fill="x")

        ttk.Separator(live_right, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(live_right, text="Last detection", font=("Segoe UI", 12, "bold")).pack(pady=(0, 6))
        self.live_result_var = tk.StringVar(value="None")
        ttk.Label(live_right, textvariable=self.live_result_var, font=("Segoe UI", 11)).pack(pady=(0, 10))

        self.live_result_image_label = ttk.Label(live_right)
        self.live_result_image_label.pack(pady=(0, 10))

        self.live_counts_var = tk.StringVar(value="")
        ttk.Label(live_right, textvariable=self.live_counts_var, font=("Consolas", 10)).pack(pady=(6, 0))

        # ======================
        # PICTURE FRAME
        # ======================
        
        self.pic_frame = ttk.Frame(self.container)

        pic_top = ttk.Frame(self.pic_frame)
        pic_top.pack(fill="x")

        ttk.Label(pic_top, text="Picture Detection", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.pic_back_btn = ttk.Button(pic_top, text="Return to Menu", command=self._leave_picture_to_menu)
        self.pic_back_btn.pack(side="right")

        pic_body = ttk.Frame(self.pic_frame)
        pic_body.pack(fill="both", expand=True, pady=(10, 0))

        left = ttk.Frame(pic_body)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(pic_body, width=340)
        right.pack(side="right", fill="y", padx=(12, 0))

        self.pic_image_label = ttk.Label(left, text="Load an image to detect trash.")
        self.pic_image_label.pack(fill="both", expand=True)

        pic_controls = ttk.Frame(left, padding=(0, 10, 0, 0))
        pic_controls.pack(fill="x")

        self.open_pic_btn = ttk.Button(pic_controls, text="Open Picture...", command=self.open_picture)
        self.open_pic_btn.pack(side="left")

        self.pic_status_var = tk.StringVar(value="Ready.")
        ttk.Label(pic_controls, textvariable=self.pic_status_var).pack(side="left", padx=12)

        ttk.Label(right, text="Report", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))
        self.pic_result_var = tk.StringVar(value="No report yet.")
        ttk.Label(right, textvariable=self.pic_result_var, font=("Segoe UI", 12)).pack(pady=(0, 10))

        self.pic_result_image_label = ttk.Label(right)
        self.pic_result_image_label.pack(pady=(0, 10))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=12)

        self.feedback_title = ttk.Label(right, text="Was it correct?", font=("Segoe UI", 11, "bold"))
        self.feedback_title.pack(pady=(0, 8))

        fb = ttk.Frame(right)
        fb.pack()

        self.yes_btn = ttk.Button(fb, text="Yes", command=self._feedback_yes, state="disabled")
        self.yes_btn.pack(side="left", padx=5)

        self.no_btn = ttk.Button(fb, text="No", command=self._feedback_no, state="disabled")
        self.no_btn.pack(side="left", padx=5)

    # ==============================
    # NAVIGATION
    # ==============================
    
    def _hide_all_frames(self):
        for f in (self.menu_frame, self.live_frame, self.pic_frame):
            f.pack_forget()

    def show_menu(self):
        self._hide_all_frames()
        self.menu_frame.pack(fill="both", expand=True)

    def show_live(self):
        self._hide_all_frames()
        self.live_frame.pack(fill="both", expand=True)
        self._start_live_detection()

    def show_picture(self):
        self._hide_all_frames()
        self.pic_frame.pack(fill="both", expand=True)
        self._stop_live_detection()

        self.pic_status_var.set("Ready.")
        self.pic_result_var.set("No report yet.")
        self.pic_result_image_label.configure(image="", text="")
        self.pic_image_label.configure(image="", text="Load an image to detect trash.")
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")
        self.last_sample = None

    def _leave_live_to_menu(self):
        self._stop_live_detection()
        self.show_menu()

    def _leave_picture_to_menu(self):
        self.show_menu()

    # ==============================
    # DRAWING
    # ==============================
    
    @staticmethod
    def _draw_fancy_bbox(frame_bgr, xyxy, label_text=""):
        x1, y1, x2, y2 = xyxy

        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        t = time.time()
        pulse = 1.0 + 0.35 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 1.5 * t))
        thick = int(round(3 * pulse))
        thick = max(2, min(thick, 6))

        neon1 = (255, 255, 0)
        neon2 = (0, 255, 255)
        shadow = (0, 0, 0)

        cv2.rectangle(frame_bgr, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), shadow, thick + 4)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), neon1, thick + 2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), neon2, thick)

        corner_len = max(18, int(0.08 * max(x2 - x1, y2 - y1)))
        corner_len = min(corner_len, 60)

        cv2.line(frame_bgr, (x1, y1), (x1 + corner_len, y1), neon2, thick + 1)
        cv2.line(frame_bgr, (x1, y1), (x1, y1 + corner_len), neon2, thick + 1)

        cv2.line(frame_bgr, (x2, y1), (x2 - corner_len, y1), neon2, thick + 1)
        cv2.line(frame_bgr, (x2, y1), (x2, y1 + corner_len), neon2, thick + 1)

        cv2.line(frame_bgr, (x1, y2), (x1 + corner_len, y2), neon2, thick + 1)
        cv2.line(frame_bgr, (x1, y2), (x1, y2 - corner_len), neon2, thick + 1)

        cv2.line(frame_bgr, (x2, y2), (x2 - corner_len, y2), neon2, thick + 1)
        cv2.line(frame_bgr, (x2, y2), (x2, y2 - corner_len), neon2, thick + 1)

        if label_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.65
            pad = 6
            (tw, th), base = cv2.getTextSize(label_text, font, fs, 2)

            lx1 = x1
            ly1 = max(0, y1 - (th + base + pad * 2) - 6)
            lx2 = min(w - 1, x1 + tw + pad * 2)
            ly2 = min(h - 1, ly1 + th + base + pad * 2)

            cv2.rectangle(frame_bgr, (lx1 + 2, ly1 + 2), (lx2 + 2, ly2 + 2), shadow, -1)
            cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), neon2, -1)

            tx = lx1 + pad
            ty = ly2 - pad - max(0, base // 2)
            cv2.putText(frame_bgr, label_text, (tx, ty), font, fs, (0, 0, 0), 2, cv2.LINE_AA)

        return frame_bgr

    @staticmethod
    def _sanitize_xyxy(xyxy, w: int, h: int):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    # ==============================
    # CAMERA PREVIEW LOOP (ALWAYS ON)
    # ==============================
    
    def _show_frame_on_live_ui(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((740, 555), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.live_preview_label.configure(image=self.preview_photo, text="")

    def _update_camera_preview(self):
        ret, frame = self.cap.read()
        if ret:
            with self.latest_frame_lock:
                self.latest_frame_bgr = frame.copy()

            if self.live_frame.winfo_ismapped():
                if self.live_freeze and self.live_frozen_frame_bgr is not None:
                    self._show_frame_on_live_ui(self.live_frozen_frame_bgr)
                else:
                    if self.live_mode:
                        with self._overlay_lock:
                            box = self.live_box
                            lbl = self.live_label
                            seen = self.live_seen_ts
                        if box is not None and (time.time() - seen) < 0.35:
                            frame = self._draw_fancy_bbox(frame, box, lbl)

                    self._show_frame_on_live_ui(frame)

        self.root.after(15, self._update_camera_preview)

    # ==============================
    # REAL-TIME DETECTION
    # ==============================
    
    def _start_live_detection(self):
        if self.live_mode:
            return

        self.live_mode = True
        self.live_status_var.set("Running real-time detection...")
        self.live_result_var.set("None")
        self.live_counts_var.set("")
        self.live_result_image_label.configure(image="", text="")

        self.live_freeze = False
        self.live_frozen_frame_bgr = None
        self.resume_btn.config(state="disabled")

        with self._overlay_lock:
            self.live_box = None
            self.live_label = ""
            self.live_conf = 0.0
            self.live_seen_ts = 0.0

        self._live_thread = threading.Thread(target=self._live_detect_loop, daemon=True)
        self._live_thread.start()

    def _stop_live_detection(self):
        self.live_mode = False
        self.live_status_var.set("Ready.")

        self.live_freeze = False
        self.live_frozen_frame_bgr = None
        self.resume_btn.config(state="disabled")

        with self._overlay_lock:
            self.live_box = None
            self.live_label = ""
            self.live_conf = 0.0
            self.live_seen_ts = 0.0

    def _live_detect_loop(self):
        while self.live_mode:
            if self.live_freeze:
                time.sleep(0.05)
                continue

            with self.latest_frame_lock:
                frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            try:
                results = self.model(frame, conf=0.25, imgsz=640, verbose=False)
            except Exception:
                time.sleep(0.05)
                continue

            best = None 
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = max(r.boxes, key=lambda bb: float(bb.conf[0]))
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                xyxy_raw = b.xyxy[0].tolist()
                x1, y1, x2, y2 = self._sanitize_xyxy(xyxy_raw, frame.shape[1], frame.shape[0])
                best = (conf, cls_id, [x1, y1, x2, y2])
                break

            if best is not None:
                conf, cls_id, xyxy = best
                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                    with self._overlay_lock:
                        self.live_box = xyxy
                        self.live_conf = conf
                        self.live_label = f"{cls_name}  {conf:.2f}"
                        self.live_seen_ts = time.time()

                    def ui_update():
                        self.live_result_var.set(f"{cls_name}  ({conf:.2f})")
                        if self.class_imgs.get(cls_name) is not None:
                            self.class_photo = self.class_imgs[cls_name]
                            self.live_result_image_label.configure(image=self.class_photo, text="")
                        else:
                            self.live_result_image_label.configure(image="", text="(No reference image)")
                    self.root.after(0, ui_update)

            time.sleep(0.05)

    # ==============================
    # BEST SCAN RESULT
    # ==============================
    
    def start_best_scan_3s(self):
        if self.running_best_scan:
            return

        self.running_best_scan = True
        self.best_scan_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")

        self.live_freeze = False
        self.live_frozen_frame_bgr = None

        self.live_status_var.set("Scanning for 3 seconds...")
        self.live_counts_var.set("")
        self.live_result_var.set("Scanning...")

        with self._overlay_lock:
            self.live_box = None
            self.live_label = ""
            self.live_conf = 0.0
            self.live_seen_ts = 0.0

        t = threading.Thread(target=self._best_scan_worker, daemon=True)
        t.start()

    def _best_scan_worker(self):
        start_time = time.time()
        counts = Counter()

        best_any = None
        best_any_conf = -1.0

        best_per_class = {c: {"conf": -1.0, "frame": None, "xyxy": None} for c in CLASS_NAMES}

        while time.time() - start_time < 3.0:
            with self.latest_frame_lock:
                frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                results = self.model(frame, conf=0.25, imgsz=640, verbose=False)
            except Exception:
                time.sleep(0.02)
                continue

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])

                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                    counts[cls_name] += 1

                    xyxy_raw = best_box.xyxy[0].tolist()
                    x1, y1, x2, y2 = self._sanitize_xyxy(xyxy_raw, frame.shape[1], frame.shape[0])
                    xyxy = [x1, y1, x2, y2]

                    with self._overlay_lock:
                        self.live_box = xyxy
                        self.live_conf = conf
                        self.live_label = f"{cls_name}  {conf:.2f}"
                        self.live_seen_ts = time.time()

                    if conf > best_any_conf:
                        best_any_conf = conf
                        best_any = (frame.copy(), xyxy, cls_name, conf)

                    if conf > best_per_class[cls_name]["conf"]:
                        best_per_class[cls_name] = {
                            "conf": conf,
                            "frame": frame.copy(),
                            "xyxy": xyxy
                        }
                break

            time.sleep(0.01)

        if counts:
            top_class, _ = counts.most_common(1)[0]
            report = f"Best Scan: {top_class} (most frequent)"

            sample = best_per_class.get(top_class)
            if sample and sample["frame"] is not None:
                self.last_sample = {
                    "predicted": top_class,
                    "frame": sample["frame"],
                    "xyxy": sample["xyxy"],
                    "conf": float(sample["conf"])
                }
            elif best_any is not None:
                frame_bgr, xyxy, cls_name, conf = best_any
                self.last_sample = {
                    "predicted": cls_name,
                    "frame": frame_bgr,
                    "xyxy": xyxy,
                    "conf": float(conf)
                }
        else:
            top_class = None
            report = "Best Scan: No detection in 3 seconds."
            self.last_sample = None

        counts_lines = [f"{c:<7}: {counts.get(c, 0)}" for c in CLASS_NAMES]
        counts_text = "\n".join(counts_lines)

        frozen = None
        if self.last_sample is not None:
            frame = self.last_sample["frame"].copy()
            x1, y1, x2, y2 = self._sanitize_xyxy(self.last_sample["xyxy"], frame.shape[1], frame.shape[0])
            label = f'{self.last_sample["predicted"]}  {self.last_sample["conf"]:.2f}'
            frozen = self._draw_fancy_bbox(frame, [x1, y1, x2, y2], label)

        def finalize_ui():
            self.live_counts_var.set(counts_text)
            self.live_result_var.set(report)

            if top_class and self.class_imgs.get(top_class) is not None:
                self.class_photo = self.class_imgs[top_class]
                self.live_result_image_label.configure(image=self.class_photo, text="")
            elif top_class:
                self.live_result_image_label.configure(image="", text="(No reference image)")
            else:
                self.live_result_image_label.configure(image="", text="")

            if frozen is not None:
                self.live_frozen_frame_bgr = frozen
                self.live_freeze = True
                self.resume_btn.config(state="normal")
            else:
                self.live_frozen_frame_bgr = None
                self.live_freeze = False
                self.resume_btn.config(state="disabled")

            with self._overlay_lock:
                self.live_box = None
                self.live_label = ""
                self.live_conf = 0.0
                self.live_seen_ts = 0.0

            self.live_status_var.set("Ready.")
            self.best_scan_btn.config(state="normal")
            self.running_best_scan = False

        self.root.after(0, finalize_ui)

    def resume_live(self):
        self.live_freeze = False
        self.live_frozen_frame_bgr = None
        self.resume_btn.config(state="disabled")
        self.live_status_var.set("Running real-time detection...")

    # ==============================
    # PICTURE DETECTION
    # ==============================
    
    def open_picture(self):
        filetypes = [
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if not path:
            return

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return

        self.pic_status_var.set("Detecting...")
        self.root.update_idletasks()

        try:
            results = self.model(img_bgr, conf=0.25, imgsz=640, verbose=False)
        except Exception as e:
            self.pic_status_var.set("Ready.")
            messagebox.showerror("Model error", str(e))
            return

        best = None 
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = max(r.boxes, key=lambda bb: float(bb.conf[0]))
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            xyxy_raw = b.xyxy[0].tolist()
            x1, y1, x2, y2 = self._sanitize_xyxy(xyxy_raw, img_bgr.shape[1], img_bgr.shape[0])
            best = (conf, cls_id, [x1, y1, x2, y2])
            break

        if best is None or not (0 <= best[1] < len(CLASS_NAMES)):
            self.pic_result_var.set("No detection found.")
            self.pic_result_image_label.configure(image="", text="")
            self._show_picture_on_ui(img_bgr)
            self.pic_status_var.set("Ready.")
            self.yes_btn.config(state="disabled")
            self.no_btn.config(state="disabled")
            self.last_sample = None
            return

        conf, cls_id, xyxy = best
        cls_name = CLASS_NAMES[cls_id]
        labeled = img_bgr.copy()
        labeled = self._draw_fancy_bbox(labeled, xyxy, f"{cls_name}  {conf:.2f}")

        self.pic_result_var.set(f"Detected: {cls_name}  ({conf:.2f})")
        if self.class_imgs.get(cls_name) is not None:
            self.class_photo = self.class_imgs[cls_name]
            self.pic_result_image_label.configure(image=self.class_photo, text="")
        else:
            self.pic_result_image_label.configure(image="", text="(No reference image)")

        self._show_picture_on_ui(labeled)
        self.pic_status_var.set("Ready.")

        self.last_sample = {
            "predicted": cls_name,
            "frame": img_bgr.copy(),
            "xyxy": xyxy,
            "conf": float(conf),
        }
        self.yes_btn.config(state="normal")
        self.no_btn.config(state="normal")

    def _show_picture_on_ui(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((740, 555), Image.LANCZOS)
        self.pic_photo = ImageTk.PhotoImage(img)
        self.pic_image_label.configure(image=self.pic_photo, text="")

    # ==============================
    # FEEDBACK (SAVES CROPS)
    # ==============================
    
    def _feedback_yes(self):
        self.pic_status_var.set("Feedback received: correct ✅")
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")

    def _feedback_no(self):
        if self.last_sample is None:
            return

        win = tk.Toplevel(self.root)
        win.title("Correct Label")
        win.geometry("320x170")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select the correct class:", font=("Segoe UI", 10, "bold")).pack(pady=(14, 8))

        sel = tk.StringVar(value=CLASS_NAMES[0] if CLASS_NAMES else "")
        combo = ttk.Combobox(win, textvariable=sel, values=CLASS_NAMES, state="readonly")
        combo.pack(pady=(0, 12))

        btns = ttk.Frame(win)
        btns.pack(pady=10)

        def save_and_close():
            correct = sel.get().strip()
            if not correct:
                messagebox.showerror("Error", "Please select a class.")
                return
            try:
                saved = self._save_feedback_sample(correct_label=correct)
                self.pic_status_var.set(f"Saved feedback sample ✅ ({saved})")
                self.yes_btn.config(state="disabled")
                self.no_btn.config(state="disabled")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

        ttk.Button(btns, text="Save", command=save_and_close).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=6)

    def _save_feedback_sample(self, correct_label: str) -> str:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pred = self.last_sample["predicted"]
        conf = self.last_sample["conf"]
        frame = self.last_sample["frame"]
        x1, y1, x2, y2 = self._sanitize_xyxy(self.last_sample["xyxy"], frame.shape[1], frame.shape[0])

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            crop = frame.copy()

        class_dir = os.path.join(FEEDBACK_DIR, correct_label)
        os.makedirs(class_dir, exist_ok=True)

        crop_path = os.path.join(class_dir, f"{ts}_crop.jpg")
        full_path = os.path.join(class_dir, f"{ts}_full.jpg")
        meta_path = os.path.join(class_dir, f"{ts}_meta.json")

        cv2.imwrite(crop_path, crop)

        vis = frame.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"pred={pred} ({conf:.2f})",
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.imwrite(full_path, vis)

        meta = {
            "timestamp": ts,
            "predicted": pred,
            "pred_conf": conf,
            "correct_label": correct_label,
            "bbox_xyxy": [x1, y1, x2, y2],
            "model_path": MODEL_PATH,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return os.path.relpath(crop_path, FEEDBACK_DIR)

    # ==============================
    # CLOSE
    # ==============================
    
    def on_close(self):
        try:
            self.live_mode = False
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TrashDetectorApp(root)
    root.mainloop()
