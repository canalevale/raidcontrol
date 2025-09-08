#!/usr/bin/env python3
import os, time
import cv2
import numpy as np
import yaml
from NumberOCR import DigitReaderSVM

# =========================
# Config
# =========================
DEFAULT_CFG = {
    "model_ocr_path": "./models/OCR/svm_ocr_arialv2.pkl",
    # Geometría del cartel ~17x8
    "aspect_target": 17.0 / 8.0,
    "aspect_tol": 0.40,      # ±40%
    "min_area_frac": 0.01,   # área mínima vs imagen (lores)
    "max_area_frac": 0.50,   # área máxima vs imagen (lores)
    # Kernels morfológicos
    "canny_values": [0, 255],
    "blur_kernel": [3, 3],
    "close_kernel": [3, 3]
}

def load_config(path="config.yaml"):
    cfg = DEFAULT_CFG.copy()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        print(f"[INFO] Config loaded from {path}")
    else:
        print("[INFO] No config.yaml found. Using defaults.")
    return cfg

CFG = load_config()

# =========================
# Utils
# =========================
def detect_race_plate(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    blurred = cv2.GaussianBlur(gray, tuple(CFG["blur_kernel"]), 0)
    _, adaptive = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    # Edge detection
    edges = cv2.Canny(adaptive, CFG["canny_values"][0], CFG["canny_values"][1])
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
    selected_contours = []
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(CFG["close_kernel"]))
    for cnt in contours:
        adaptive2 = None
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)/(frame.shape[0]*frame.shape[1])
        if area > 0.01:
            print(f"Aspect: {aspect_ratio:.2f}, Area: {area:.4f}")
        if 0.8 < aspect_ratio < 3.0 and 0.001 < area < 0.05:
            print(area)
            selected_contours.append((x, y, w, h))
            plate_crop = gray[y:y+h, x:x+w]
            cv2.imshow("crops", plate_crop)
            _, adaptive2 = cv2.threshold(plate_crop, 150, 255, cv2.THRESH_BINARY)
            
            adaptive2 = cv2.morphologyEx(adaptive2, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("Plate", adaptive2)
            break
        
    if debug:
        cv2.imshow("Threshold", adaptive)
        cv2.imshow("Edges", edges)
        
        
    if adaptive2 is not None:
        return adaptive2
    else:
        return adaptive

# =========================
# Loop
# =========================
debug = CFG.get("draw_debug", False)
try:
    reader = DigitReaderSVM(svm_model_path=CFG['model_ocr_path'],binarize_inverted=False)
    frame = cv2.imread("examples/bike_2025-06-18_16-38-06_853020.jpg")
    processed_frame = detect_race_plate(frame)

    number, digits = reader.read_number(processed_frame)
    print(f"Detected number: {number} from digits {digits}")
    while True:
        cv2.imshow(f"Detection:{number}", processed_frame)
        if  cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    try: cv2.destroyAllWindows()
    except: pass
