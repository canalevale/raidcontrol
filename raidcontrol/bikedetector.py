import os
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from picamera2 import Picamera2
from libcamera import Transform
from ultralytics import YOLO

# =========================
# Config
# =========================
DEFAULT_CONFIG = {
    "save_base": "clips2",
    "model_path": "./yolov8n_ncnn_model",
    "detect_class": "bicycle",
    "min_box_w": 40,
    "min_box_h": 40,
    "save_crop_on_cross": True,
    "main_size": [4656, 3496],
    "lores_size": [864, 480],
    "line_y_ratio": 0.75,  
    "draw_debug": True,
    "window_name": "lores_debug",
    "camera_transform": {"vflip": True, "hflip": True},
    "camera_controls": {"AfMode": 2, "AfTrigger": 0, "AfSpeed": 1, "ExposureTime": 0, "AnalogueGain": 10},
    "inference": {"imgsz": 320, "conf": 0.35},
}

def load_config(path="config.yaml"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg.update(data)
        print(f"[INFO] Config loaded from: {path}")
    else:
        print("[INFO] No config.yaml found. Using defaults.")

    cfg["main_size"] = tuple(cfg.get("main_size", DEFAULT_CONFIG["main_size"]))
    cfg["lores_size"] = tuple(cfg.get("lores_size", DEFAULT_CONFIG["lores_size"]))
    cfg["line_y_ratio"] = float(cfg.get("line_y_ratio", DEFAULT_CONFIG["line_y_ratio"]))
    cfg["line_y_ratio"] = max(0.0, min(1.0, cfg["line_y_ratio"]))
    inf = cfg.get("inference", {})
    cfg["inference"] = {"imgsz": int(inf.get("imgsz", 320)), "conf": float(inf.get("conf", 0.35))}
    cfg["min_box_w"] = int(cfg.get("min_box_w", 40))
    cfg["min_box_h"] = int(cfg.get("min_box_h", 40))
    cfg["save_crop_on_cross"] = bool(cfg.get("save_crop_on_cross", True))
    cfg["draw_debug"] = bool(cfg.get("draw_debug", True))
    return cfg

CFG = load_config()

# =========================
# Paths
# =========================
RUN_DIR = os.path.join(CFG["save_base"], f"crops")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"[INFO] Saving outputs in: {RUN_DIR}")

# =========================
# Utils
# =========================
def clamp_box(x1, y1, x2, y2, w, h):
    return (
        max(0, min(int(x1), w - 1)),
        max(0, min(int(y1), h - 1)),
        max(0, min(int(x2), w - 1)),
        max(0, min(int(y2), h - 1)),
    )

def bbox_bottom_center(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cx, cy


# =========================
# Model
# =========================
model = YOLO(CFG["model_path"])
names = getattr(model, "names", {})

try:
    TARGET_IDX = next(k for k, v in names.items() if v == CFG["detect_class"])
except StopIteration:
    print(f"[WARN] Class '{CFG['detect_class']}' not found in model names. Using ALL classes.")
    TARGET_IDX = None

# =========================
# Camera
# =========================
picam2 = Picamera2()
transform = Transform(vflip=CFG["camera_transform"]["vflip"], hflip=CFG["camera_transform"]["hflip"])
config = picam2.create_preview_configuration(
    main={"size": CFG["main_size"], "format": "RGB888"},
    lores={"size": CFG["lores_size"], "format": "RGB888"},
    transform=transform,
)
picam2.configure(config)
picam2.start()
try:
    picam2.set_controls(CFG["camera_controls"])
except Exception as e:
    print(f"[WARN] Could not apply some camera controls: {e}")

print("IMX519 camera started (continuous AF).")

# =========================
# State variables
# =========================
frames = 0
saves = 0

lw, lh = CFG["lores_size"]
fw, fh = CFG["main_size"]
sx, sy = fw / float(lw), fh / float(lh)

LINE_Y = int(CFG["lores_size"][1] * CFG["line_y_ratio"])  # single horizontal line across full width

def shutdown(t0):
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"\nShutting down. Frames: {frames}, Saves: {saves}, Elapsed: {elapsed:.1f}s, ~{fps:.2f} FPS")
    try: picam2.stop()
    except Exception: pass
    if CFG["draw_debug"]:
        try: cv2.destroyAllWindows()
        except Exception: pass

# =========================
# Main
# =========================
t0 = time.time()
prev_dets = []  
try:
    while True:
        frame , _ = picam2.capture_arrays(["lores", "main"])
        frames += 1
        lores = frame[0]
        full = frame[1]
        if TARGET_IDX is not None:

            results = model(lores, imgsz=CFG["inference"]["imgsz"], conf=CFG["inference"]["conf"],
                            classes=[TARGET_IDX], verbose=False)
        else:
            results = model(lores, imgsz=CFG["inference"]["imgsz"], conf=CFG["inference"]["conf"], verbose=False)

        # Draw single line for debug
        if CFG["draw_debug"]:
            cv2.line(lores, (0, LINE_Y), (lw-1, LINE_Y), (0, 255, 0), 2)

        curr_dets = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1l, y1l, x2l, y2l = map(int, boxes.xyxy[i].tolist())

                # bottom-center
                cx = (x1l + x2l) // 2
                cy = y2l
                # distance for line
                dist_to_line = cy - LINE_Y
                # full-res bbox
                x1f = int(x1l * sx); y1f = int(y1l * sy)
                x2f = int(x2l * sx); y2f = int(y2l * sy)
                x1f, y1f, x2f, y2f = clamp_box(x1f, y1f, x2f, y2f, fw, fh)
                if CFG["save_crop_on_cross"] and dist_to_line > 0:
                    bw, bh = x2f - x1f, y2f - y1f
                    if bw >= CFG["min_box_w"] and bh >= CFG["min_box_h"]:
                        crop_rgb = full[y1f:y2f, x1f:x2f]
                        if crop_rgb.size > 0:
                            # filename with wall-clock timestamp, and pixel distance
                            ts_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
                            fname = f"cross_{ts_str}_d{dist_to_line:+d}px.jpg"
                            fpath = os.path.join(RUN_DIR, fname)
                            if cv2.imwrite(fpath, crop_rgb):
                                saves += 1
                                print(f"[SAVE] {fpath}")
                if CFG["draw_debug"]:
                    color = (0, 255, 0) if dist_to_line < 0 else ((0, 165, 255) if dist_to_line == 0 else (0, 0, 255))
                    cv2.rectangle(lores, (x1l, y1l), (x2l, y2l), color, 2)
                    cv2.circle(lores, (cx, cy), 4, (0, 0, 255), -1)
                   
        # --- Debug window / exit ---
        if CFG["draw_debug"]:
            cv2.imshow(CFG["window_name"], lores)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

except KeyboardInterrupt:
    pass
finally:
    shutdown(t0)