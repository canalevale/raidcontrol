import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort


# ---------------- Preprocess ----------------

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to a padded rectangle while keeping aspect ratio (YOLO-style letterbox).
    Returns: (img_letterboxed, scale_ratio, (pad_x, pad_y))
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))

    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_y,
        pad_h - pad_y,
        pad_x,
        pad_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return img_padded, r, (pad_x, pad_y)


# ---------------- NMS ----------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """
    Pure numpy NMS.
    boxes: [N,4] in xyxy
    scores: [N]
    return: kept indices
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


# ---------------- Output helpers ----------------

def to_NC(output: np.ndarray) -> np.ndarray:
    """
    Convert output to shape (N, C) where C = 4 + num_classes.
    Supports common YOLOv8 ONNX outputs:
      - (1, C, N)
      - (1, N, C)
      - (C, N)
      - (N, C)
    """
    out = np.squeeze(output)

    if out.ndim != 2:
        raise RuntimeError(f"Unexpected output dims: {output.shape} -> {out.shape}")

    # C is usually small (e.g., 5..200), N is large (thousands)
    if out.shape[0] < out.shape[1]:
        # (C, N) -> (N, C)
        out = out.T

    return out  # (N, C)


def infer_num_classes_from_output(out0: np.ndarray) -> int:
    """
    Infer nc from output tensor.
    Works for typical shapes like (1,84,8400) or (1,8400,84).
    """
    out = np.squeeze(out0)
    if out.ndim != 2:
        raise RuntimeError(f"Cannot infer classes from output shape: {out0.shape}")

    c = min(out.shape[0], out.shape[1])  # channels is the small dimension
    if c <= 4:
        raise RuntimeError(f"Cannot infer classes (C<=4). Output shape: {out0.shape}")
    return int(c - 4)


# ---------------- Postprocess ----------------

def postprocess_yolov8(
    out0: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    num_classes: int,
    imgsz: int,
) -> np.ndarray:
    """
    Postprocess common YOLOv8 ONNX output:
    - Convert to (N, C), then parse [xywh + class scores]
    - Auto-scale xywh if normalized (0..1)
    - NMS
    Returns dets: [M, 6] => xyxy, score, class_id
    """
    out = to_NC(out0)  # (N, C)
    if out.shape[1] < 4 + num_classes:
        raise RuntimeError(f"Output channels too small: got {out.shape[1]}, expected {4+num_classes}")

    xywh = out[:, 0:4].astype(np.float32)
    cls_scores = out[:, 4:4 + num_classes].astype(np.float32)

    # Some exports output normalized coords (0..1). If so, scale to imgsz.
    if float(np.max(xywh)) <= 1.5:
        xywh *= float(imgsz)

    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    mask = scores >= conf_thres
    xywh = xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if xywh.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # xywh -> xyxy
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, scores, iou_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    dets = np.concatenate(
        [boxes, scores.reshape(-1, 1), class_ids.reshape(-1, 1).astype(np.float32)],
        axis=1
    )
    return dets.astype(np.float32)


# ---------------- Pretty drawing ----------------

def class_color_bgr(class_id: int) -> Tuple[int, int, int]:
    """
    Deterministic nice-ish color per class (BGR for OpenCV).
    """
    # A small curated palette (BGR) and fallback hashing.
    palette = [
        (255, 56, 56),   # red-ish
        (255, 157, 151), # salmon
        (255, 112, 31),  # orange
        (255, 178, 29),  # amber
        (207, 210, 49),  # yellow-green
        (72, 249, 10),   # green
        (146, 204, 23),  # leaf
        (61, 219, 134),  # teal
        (26, 147, 52),   # dark green
        (0, 212, 187),   # cyan
        (44, 153, 168),  # blue-green
        (0, 194, 255),   # light blue
        (52, 69, 147),   # indigo
        (100, 115, 255), # periwinkle
        (0, 24, 236),    # strong blue
        (132, 56, 255),  # purple
        (82, 0, 133),    # deep purple
        (203, 56, 255),  # magenta
        (255, 149, 200), # pink
        (255, 55, 199),  # hot pink
    ]
    if class_id < len(palette):
        return palette[class_id]
    # fallback: hash -> color, keep it vivid
    rng = np.random.default_rng(class_id * 99991)
    c = rng.integers(60, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_label_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 3
) -> None:
    """
    Draw a pretty bounding box with a label background.
    """
    # Outer box (colored) with a black outline for contrast
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    txt_th = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, txt_th)

    # Place label at top-left of bbox (if no space above, put inside)
    pad = 4
    y_text_top = y1 - th - 2 * pad
    if y_text_top < 0:
        y_text_top = y1 + 2  # inside box

    # Background rectangle
    x_bg1 = x1
    y_bg1 = y_text_top
    x_bg2 = x1 + tw + 2 * pad
    y_bg2 = y_text_top + th + 2 * pad

    # Background with black border
    cv2.rectangle(img, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x_bg1, y_bg1), (x_bg2, y_bg2), color, -1)
    # slight transparency
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    # Text (white)
    x_text = x1 + pad
    y_text = y_bg1 + th + pad
    cv2.putText(img, label, (x_text, y_text), font, font_scale, (255, 255, 255), txt_th, cv2.LINE_AA)


def draw_dets_pretty(
    img: np.ndarray,
    dets: np.ndarray,
    r: float,
    pad_x: int,
    pad_y: int,
    names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    Draw detections on original image (undo letterbox) with nice colored boxes per class.
    """
    out = img.copy()
    h, w = out.shape[:2]

    for x1, y1, x2, y2, score, cls_id_f in dets:
        cls_id = int(cls_id_f)

        # Undo letterbox
        x1 = (x1 - pad_x) / r
        y1 = (y1 - pad_y) / r
        x2 = (x2 - pad_x) / r
        y2 = (y2 - pad_y) / r

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        color = class_color_bgr(cls_id)
        cname = names.get(cls_id, f"id={cls_id}") if names else f"id={cls_id}"
        label = f"{cname} {score:.2f}"

        # thickness based on image size
        thickness = max(2, int(round(0.0025 * (h + w) / 2)))
        draw_label_box(out, x1, y1, x2, y2, label, color, thickness=thickness)

    return out


# ---------------- Names loading ----------------

def load_class_names(path: Path) -> Dict[int, str]:
    """
    Load class names from:
      - YOLO data.yaml-like file containing 'names:'
      - Plain text: one class name per line
    No external YAML dependency required.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Try simple "names:" parsing
    names: Dict[int, str] = {}
    if "names:" in text:
        # naive parse: look for lines after "names:" like "  0: class"
        lines = text.splitlines()
        in_names = False
        for line in lines:
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                s = line.strip()
                if not s:
                    continue
                # stop if next top-level key appears
                if ":" in s and not s[0].isdigit() and not s.startswith("-"):
                    # e.g. "train:" or "val:" keys
                    break
                # formats:
                # "0: numero"
                # "- numero"
                if s[0].isdigit() and ":" in s:
                    k, v = s.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    try:
                        names[int(k)] = v
                    except ValueError:
                        pass
                elif s.startswith("-"):
                    # list format
                    v = s.lstrip("-").strip().strip("'").strip('"')
                    names[len(names)] = v

        if names:
            return names

    # Fallback: treat as txt list
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        names[i] = ln
    return names


# ---------------- IO ----------------

def list_images(input_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        return [input_path]
    imgs = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best.onnx", help="Path to ONNX model")
    ap.add_argument("--input", required=True, help="Image file or folder path")
    ap.add_argument("--outdir", default="onnx_out", help="Output directory for annotated images")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (should match export/training)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    ap.add_argument("--nc", type=int, default=-1, help="Number of classes. -1 = infer from ONNX output")
    ap.add_argument("--names", default="", help="Path to data.yaml or names.txt to show class names")
    ap.add_argument("--debug", action="store_true", help="Print debug info about ONNX output")
    ap.add_argument("--show", action="store_true", help="Show window (requires GUI)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = None
    if args.names:
        names = load_class_names(Path(args.names))
        print(f"[INFO] Loaded {len(names)} class names from {args.names}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    images = list_images(input_path)
    if not images:
        raise SystemExit(f"No images found in: {input_path}")

    num_classes = None if args.nc == -1 else args.nc
    printed_debug = False

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        img_lb, r, (pad_x, pad_y) = letterbox(img, (args.imgsz, args.imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, axis=0)   # add batch

        outputs = sess.run(None, {input_name: x})
        out0 = outputs[0]

        if args.debug and not printed_debug:
            tmp = np.squeeze(out0)
            print("[DEBUG] out0 shape:", out0.shape)
            print("[DEBUG] squeezed shape:", tmp.shape, "min/max:", float(tmp.min()), float(tmp.max()))
            printed_debug = True

        if num_classes is None:
            num_classes = infer_num_classes_from_output(out0)
            print(f"[INFO] Inferred num_classes={num_classes} from output shape {out0.shape}")

        dets = postprocess_yolov8(out0, args.conf, args.iou, num_classes, args.imgsz)
        annotated = draw_dets_pretty(img, dets, r, pad_x, pad_y, names=names)

        out_path = outdir / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        print(f"[OK] {img_path.name} -> {out_path.name} (detections: {len(dets)})")

        if args.show:
            cv2.imshow("ONNX YOLOv8 (pretty)", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
