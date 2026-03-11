import os
import sys
import yaml
import cv2
import csv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ultralytics import YOLO

# Add parent directory to path to import from raidcontrol module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from raidcontrol.NumberOCR_CNN import DigitReaderCNNONNX

import random

# Random but fixed for this execution
RUN_COLOR = (
    random.randint(50, 255),
    random.randint(50, 255),
    random.randint(50, 255),
)



# ------------------------ Utils ------------------------

def load_cfg(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_detect_classes(detect_cfg) -> Optional[List[str]]:
    if detect_cfg is None:
        return None
    if isinstance(detect_cfg, str):
        parts = [p.strip() for p in detect_cfg.split(",")]
        return parts if parts else None
    if isinstance(detect_cfg, (list, tuple)):
        return [str(p).strip() for p in detect_cfg if str(p).strip()]
    raise ValueError("detect_class must be str or list")


def resolve_target_indices(names: Dict[int, str], detect_classes: Optional[List[str]]) -> Optional[List[int]]:
    if not detect_classes:
        print("[WARN] No detect_class configured. Using ALL classes.")
        return None

    name_to_id = {v: k for k, v in names.items()}
    idxs = []

    for cls in detect_classes:
        if cls in name_to_id:
            idxs.append(name_to_id[cls])
        else:
            print(f"[WARN] Class '{cls}' not found in model names.")

    if not idxs:
        print("[WARN] No valid detect_class found. Using ALL classes.")
        return None

    print(f"[INFO] Using YOLO class filter IDs: {idxs}")
    return idxs
def draw_detection(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2
):
    """
    Draws a nice bounding box with a filled label background.
    """
    x1, y1, x2, y2 = box

    # --- Bounding box ---
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # --- Text size ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_th = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_th)

    # --- Label background ---
    pad = 4
    label_y1 = max(0, y1 - th - 2 * pad)
    label_y2 = y1
    label_x1 = x1
    label_x2 = x1 + tw + 2 * pad

    # Filled rectangle
    cv2.rectangle(
        img,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1
    )

    # --- Text ---
    cv2.putText(
        img,
        label,
        (label_x1 + pad, label_y2 - pad),
        font,
        font_scale,
        (0, 0, 0),   # black text
        font_th,
        cv2.LINE_AA
    )


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in [".mp4", ".avi", ".mov", ".mkv", ".m4v"]


# ------------------------ Frame processing ------------------------

def process_frame(
    frame: np.ndarray,
    yolo: YOLO,
    target_idxs: Optional[List[int]],
    ocr: DigitReaderCNNONNX,
    cfg: Dict[str, Any]
):
    h, w = frame.shape[:2]
    inf = cfg["inference"]

    results = yolo(
        frame,
        imgsz=int(inf["imgsz"]),
        conf=float(inf["score_th"]),
        classes=target_idxs,
        verbose=False
    )

    out = frame.copy()
    detections = []

    if not results or not results[0].boxes:
        return out, detections

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy().astype(int)
    names = yolo.names

    for (x1, y1, x2, y2), score, cls_id in zip(boxes, confs, clss):

        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            continue
        if cls_id == 1:
            crop = frame[y1:y2, x1:x2]
            number, digits, confs, plate_color = ocr.read_number(crop, bgr=True)
            label = f"Numero:{number if number is not None else 'NA'}, Color:{plate_color}"
        else:
            number = None
            digits = []
            label = f"{names[cls_id]} {score:.2f}"

        draw_detection(
            out,
            (x1, y1, x2, y2),
            label,
            color=RUN_COLOR
        )


        detections.append({
            "cls_id": int(cls_id),
            "cls_name": names[cls_id],
            "conf": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "ocr_number": number,
            "ocr_digits": digits
        })

    return out, detections


# ------------------------ Main ------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <config.yaml> <image|video> [output]")
        sys.exit(1)

    cfg = load_cfg(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    print("[INFO] Loading YOLO NCNN model...")
    yolo = YOLO("models/vueltaalpartido_v1/best_ncnn_model")

    detect_classes = normalize_detect_classes(cfg.get("detect_class"))
    target_idxs = resolve_target_indices(yolo.names, detect_classes)

    print("[INFO] Loading OCR NumPy CNN...")
    ocr = DigitReaderCNNONNX(
        onnx_path=cfg["ocr"]["model_ocr_path"],
        yaml_path=sys.argv[1],
        debug=True
    )

    # ---------- IMAGE ----------
    if is_image(input_path):
        img = cv2.imread(input_path)
        img = cv2.resize(img, tuple(cfg["main_size"]))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        out, dets = process_frame(img, yolo, target_idxs, ocr, cfg)

        print(f"[INFO] Detections: {len(dets)}")
        for d in dets:
            print(d)

        if output_path:
            cv2.imwrite(output_path, out)
            print(f"[INFO] Saved {output_path}")
        else:
            cv2.imshow(cfg["window_name"], out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ---------- VIDEO ----------
    if is_video(input_path):
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0 else 25,
                (w, h)
            )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out, _ = process_frame(frame, yolo, target_idxs, ocr, cfg)
            if writer:
                writer.write(out)
            else:
                cv2.imshow(cfg["window_name"], out)
                if cv2.waitKey(1) == 27:
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[INFO] Processed frame {frame_idx}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Video done.")
        return

    raise ValueError("Unsupported input type")


if __name__ == "__main__":
    main()
