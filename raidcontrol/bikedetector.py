#!/usr/bin/env python3
import os
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from picamera2 import Picamera2
from libcamera import Transform

import hailo_platform as hpf

from NumberOCR_CNN import DigitReaderCNNONNX


from uploader import writer_init, writer_close, write_event


# ============================================================
# Logging
# ============================================================
def get_logger(name: str = "BikeDetector") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


LOGGER = get_logger("BikeDetector")


class TimingStats:
    def __init__(self):
        self.sum: Dict[str, float] = {}
        self.count = 0

    def add(self, key: str, dt_s: float):
        self.sum[key] = self.sum.get(key, 0.0) + float(dt_s)
        self.count += 1

    def snapshot_ms(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {k: (v / self.count) * 1000.0 for k, v in self.sum.items()}

    def reset(self):
        self.sum.clear()
        self.count = 0


# =========================
# Config
# =========================
DEFAULT_CONFIG = {
    # Camera
    "main_size": [1920, 1080],
    "camera_transform": {"vflip": True, "hflip": True},
    "camera_controls": {"AfMode": 2, "AfTrigger": 0, "AfSpeed": 1, "ExposureTime": 0, "AnalogueGain": 10},

    # Line
    "line_y_ratio": 0.75,

    # Hailo
    "hef_path": "models/vueltaalpartido_v1/raid_yolo.hef",
    "output_key": "yolov8n/yolov8_nms_postprocess",
    "inference": {
        "score_th": 0.35,
        "model_channels": "RGB",
        "input_quantized_uint8": True,
        "max_dets_total": 200,
    },

    # Classes
    "class_cyclist": 0,
    "class_number": 1,

    # OCR
    "ocr": {
        "enabled": True,
        "model_ocr_path": "./ocr.onnx",
    },

    # Crossing + association
    "cross": {
        "cooldown_ms": 900,
        "key_quant": 24,
        "number_inside_only": True,
        "number_iou_fallback": True,
        "min_number_iou": 0.02,
        # Detección independiente de números
        "track_orphan_numbers": True,
        "number_cooldown_ms": 500,
        "number_key_quant": 16,
    },

    # Logging
    "logging": {
        "level": "INFO",             # DEBUG / INFO / WARNING
        "timing_every_frames": 60,   # Print perf stats every N frames
    },
}


def load_config(path="config.yaml"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg.update(data)
        LOGGER.info("Config loaded from: %s", path)
    else:
        LOGGER.info("No config.yaml found. Using defaults.")

    cfg["main_size"] = tuple(cfg.get("main_size", DEFAULT_CONFIG["main_size"]))

    cfg["line_y_ratio"] = float(cfg.get("line_y_ratio", DEFAULT_CONFIG["line_y_ratio"]))
    cfg["line_y_ratio"] = max(0.0, min(1.0, cfg["line_y_ratio"]))

    inf = cfg.get("inference", {})
    cfg["inference"] = {
        "score_th": float(inf.get("score_th", DEFAULT_CONFIG["inference"]["score_th"])),
        "model_channels": str(inf.get("model_channels", DEFAULT_CONFIG["inference"]["model_channels"])).upper(),
        "input_quantized_uint8": bool(inf.get("input_quantized_uint8", True)),
        "max_dets_total": int(inf.get("max_dets_total", 200)),
    }
    if cfg["inference"]["model_channels"] not in ("RGB", "BGR"):
        cfg["inference"]["model_channels"] = "RGB"

    cfg["class_cyclist"] = int(cfg.get("class_cyclist", DEFAULT_CONFIG["class_cyclist"]))
    cfg["class_number"] = int(cfg.get("class_number", DEFAULT_CONFIG["class_number"]))

    ocr = cfg.get("ocr", {})
    cfg["ocr"] = {
        "enabled": bool(ocr.get("enabled", True)),
        "model_ocr_path": str(ocr.get("model_ocr_path", "./ocr.onnx")),
    }

    cross = cfg.get("cross", {})
    cfg["cross"] = {
        "cooldown_ms": int(cross.get("cooldown_ms", 900)),
        "key_quant": int(cross.get("key_quant", 24)),
        "number_inside_only": bool(cross.get("number_inside_only", True)),
        "number_iou_fallback": bool(cross.get("number_iou_fallback", True)),
        "min_number_iou": float(cross.get("min_number_iou", 0.02)),
        # Parámetros para números huérfanos
        "track_orphan_numbers": bool(cross.get("track_orphan_numbers", True)),
        "number_cooldown_ms": int(cross.get("number_cooldown_ms", 500)),
        "number_key_quant": int(cross.get("number_key_quant", 16)),
    }

    logcfg = cfg.get("logging", {})
    cfg["logging"] = {
        "level": str(logcfg.get("level", "INFO")).upper(),
        "timing_every_frames": int(logcfg.get("timing_every_frames", 60)),
    }
    cfg["logging"]["timing_every_frames"] = max(1, cfg["logging"]["timing_every_frames"])

    return cfg


CFG = load_config()

# Apply log level from config
_level = getattr(logging, CFG["logging"]["level"], logging.INFO)
LOGGER.setLevel(_level)
for _h in LOGGER.handlers:
    _h.setLevel(_level)


# =========================
# Data types
# =========================
@dataclass
class Det:
    class_id: int
    score: float
    ymin: float
    xmin: float
    ymax: float
    xmax: float


# =========================
# Geometry helpers
# =========================
def clamp_box(x1, y1, x2, y2, w, h):
    return (
        max(0, min(int(x1), w - 1)),
        max(0, min(int(y1), h - 1)),
        max(0, min(int(x2), w - 1)),
        max(0, min(int(y2), h - 1)),
    )


def bbox_bottom_center(x1, y1, x2, y2) -> Tuple[int, int]:
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cx, cy


def bbox_center(x1, y1, x2, y2) -> Tuple[int, int]:
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def point_in_box(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> bool:
    return (px >= x1 and px <= x2 and py >= y1 and py <= y2)


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


# =========================
# Hailo preprocess + parse
# =========================
def preprocess_uint8(frame_rgb: np.ndarray, in_w: int, in_h: int, new_w: int, new_h: int, pad_x: int, pad_y: int, model_channels: str) -> np.ndarray:
    img = frame_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((in_h, in_w, 3), 0, dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out


def preprocess_float32(frame_rgb: np.ndarray, in_w: int, in_h: int, new_w: int, new_h: int, pad_x: int, pad_y: int, model_channels: str) -> np.ndarray:
    img = frame_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((in_h, in_w, 3), 0, dtype=np.float32)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out / 255.0


def parse_hailo_nms_by_class(nms_out, score_th: float, in_w: int, in_h: int, batch_idx: int = 0) -> List[Det]:
    if not isinstance(nms_out, list) or len(nms_out) == 0:
        return []

    per_class = nms_out[batch_idx]
    dets: List[Det] = []

    # Detect coordinate mode
    coord_mode = "px"
    for arr in per_class:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        y1, x1, y2, x2, _s = arr[0].tolist()
        mx = max(abs(float(x2)), abs(float(y2)), abs(float(x1)), abs(float(y1)))
        coord_mode = "norm" if mx <= 1.5 else "px"
        break

    for class_id, arr in enumerate(per_class):
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        for row in arr:
            y1, x1, y2, x2, s = row.tolist()
            s = float(s)
            if s < score_th:
                continue

            y1 = float(y1)
            x1 = float(x1)
            y2 = float(y2)
            x2 = float(x2)

            if coord_mode == "norm":
                y1 *= in_h
                y2 *= in_h
                x1 *= in_w
                x2 *= in_w

            dets.append(Det(
                class_id=int(class_id),
                score=s,
                ymin=y1, xmin=x1, ymax=y2, xmax=x2
            ))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def map_det_to_frame(det: Det, frame_w: int, frame_h: int, in_w: int, in_h: int, r: float, pad_x: int, pad_y: int) -> Tuple[int, int, int, int]:
    x1 = (det.xmin - pad_x) / r
    y1 = (det.ymin - pad_y) / r
    x2 = (det.xmax - pad_x) / r
    y2 = (det.ymax - pad_y) / r

    x1 = max(0, min(frame_w - 1, int(round(x1))))
    y1 = max(0, min(frame_h - 1, int(round(y1))))
    x2 = max(0, min(frame_w - 1, int(round(x2))))
    y2 = max(0, min(frame_h - 1, int(round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


# =========================
# Crossing gate (for cyclists)
# =========================
class CrossGate:
    def __init__(self, line_y: int, key_quant: int, cooldown_ms: int):
        self.line_y = int(line_y)
        self.key_quant = max(1, int(key_quant))
        self.cooldown_ms = max(0, int(cooldown_ms))

        self.last_side: Dict[Tuple[int, int, int, int], int] = {}
        self.last_cross_ts: Dict[Tuple[int, int, int, int], float] = {}

    def make_key(self, cx: int, cy_bottom: int, w: int, h: int) -> Tuple[int, int, int, int]:
        q = self.key_quant
        return (int(cx / q), int(cy_bottom / q), int(w / q), int(h / q))

    def update(self, cx: int, cy_bottom: int, w: int, h: int) -> Tuple[int, bool]:
        dist = int(cy_bottom - self.line_y)
        side = -1 if dist < 0 else 1
        key = self.make_key(cx, cy_bottom, w, h)

        prev = self.last_side.get(key, -1)
        crossed_now = (prev < 0 and side > 0)

        if crossed_now and self.cooldown_ms > 0:
            now = time.time()
            last_ts = self.last_cross_ts.get(key, 0.0)
            if (now - last_ts) * 1000.0 < self.cooldown_ms:
                crossed_now = False
            else:
                self.last_cross_ts[key] = now

        self.last_side[key] = side
        return dist, crossed_now


# =========================
# Association: cyclist -> number det
# =========================
def pick_number_for_cyclist(
    cyclist_box: Tuple[int, int, int, int],
    number_dets: List[Tuple[Tuple[int, int, int, int], float]],
    inside_only: bool,
    iou_fallback: bool,
    min_iou: float,
) -> Optional[Tuple[int, int, int, int]]:
    cx1, cy1, cx2, cy2 = cyclist_box

    best_box = None
    best_score = -1.0

    # Prefer number center inside cyclist bbox (highest score)
    for (nb, s) in number_dets:
        nx1, ny1, nx2, ny2 = nb
        nxc, nyc = bbox_center(nx1, ny1, nx2, ny2)
        if point_in_box(nxc, nyc, cx1, cy1, cx2, cy2):
            if s > best_score:
                best_score = s
                best_box = nb

    if best_box is not None:
        return best_box

    if inside_only:
        return None

    # IoU fallback
    if iou_fallback:
        best_iou = 0.0
        best_box = None
        for (nb, _s) in number_dets:
            i = iou_xyxy(cyclist_box, nb)
            if i >= min_iou and i > best_iou:
                best_iou = i
                best_box = nb
        return best_box

    return None


# =========================
# OCR
# =========================

ocr = DigitReaderCNNONNX(onnx_path=CFG["ocr"]["model_ocr_path"], yaml_path="config.yaml", debug=False)

# =========================
# Camera
# =========================
picam2 = Picamera2()
transform = Transform(vflip=CFG["camera_transform"]["vflip"], hflip=CFG["camera_transform"]["hflip"])
config = picam2.create_preview_configuration(
    main={"size": CFG["main_size"], "format": "RGB888"},
    transform=transform,
)
picam2.configure(config)
picam2.start()
try:
    picam2.set_controls(CFG["camera_controls"])
except Exception as e:
    LOGGER.warning("Could not apply some camera controls: %s", e)
LOGGER.info("Camera started.")

# =========================
# Uploader
# =========================

writer_init(cfg_path="config.yaml")

# =========================
# Main
# =========================
def main():
    fw, fh = CFG["main_size"]
    line_y = int(fh * CFG["line_y_ratio"])

    score_th = CFG["inference"]["score_th"]
    model_channels = CFG["inference"]["model_channels"]
    input_quantized_uint8 = CFG["inference"]["input_quantized_uint8"]
    max_dets_total = CFG["inference"]["max_dets_total"]

    class_cyclist = CFG["class_cyclist"]
    class_number = CFG["class_number"]

    gate = CrossGate(
        line_y=line_y,
        key_quant=CFG["cross"]["key_quant"],
        cooldown_ms=CFG["cross"]["cooldown_ms"],
    )

    # Gate para números independientes
    number_gate = None
    if CFG["cross"]["track_orphan_numbers"]:
        number_gate = CrossGate(
            line_y=line_y,
            key_quant=CFG["cross"]["number_key_quant"],
            cooldown_ms=CFG["cross"]["number_cooldown_ms"],
        )
        LOGGER.info("Orphan number tracking ENABLED")

    # Hailo init
    hef = hpf.HEF(CFG["hef_path"])
    input_infos = hef.get_input_vstream_infos()
    if len(input_infos) != 1:
        raise RuntimeError(f"Expected 1 input, got {len(input_infos)}")
    in_info = input_infos[0]
    in_h, in_w, in_c = in_info.shape
    if in_c != 3:
        raise RuntimeError(f"Expected 3 channels, got {in_c}")

    output_infos = hef.get_output_vstream_infos()
    out_names = [o.name for o in output_infos]
    out_key = CFG["output_key"] if CFG["output_key"] in out_names else out_names[0]
    if out_key != CFG["output_key"]:
        LOGGER.warning("output_key '%s' not found. Using '%s'.", CFG["output_key"], out_key)
    
    # Padding calculations 
    r = min(in_w / float(fw), in_h / float(fh))
    new_w = int(round(fw * r))
    new_h = int(round(fh * r))
    pad_x = (in_w - new_w) // 2
    pad_y = (in_h - new_h) // 2

    LOGGER.info("HEF loaded: %s", CFG["hef_path"])
    LOGGER.info("Input: %s shape=%s", in_info.name, in_info.shape)
    LOGGER.info("Output key: %s", out_key)
    LOGGER.info("Line Y (main): %d px", line_y)

    with hpf.VDevice() as vdevice:
        cfg_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = vdevice.configure(hef, cfg_params)[0]
        ng_params = network_group.create_params()

        if input_quantized_uint8:
            in_params = hpf.InputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=hpf.FormatType.UINT8
            )
        else:
            in_params = hpf.InputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
            )

        out_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        try:
            with network_group.activate(ng_params):
                with hpf.InferVStreams(network_group, in_params, out_params) as pipe:
                    while True:
                        # 1) Capture
                        frame_rgb = picam2.capture_array("main")  # RGB

                        # 2) Preprocess
                        if input_quantized_uint8:
                            x = preprocess_uint8(frame_rgb, in_w, in_h, new_w, new_h, pad_x, pad_y, model_channels=model_channels)
                        else:
                            x = preprocess_float32(frame_rgb, in_w, in_h, new_w, new_h, pad_x, pad_y, model_channels=model_channels)
                        input_dict = {in_info.name: np.expand_dims(x, axis=0)}

                        # 3) Inference
                        infer_results = pipe.infer(input_dict)
                        nms_out = infer_results[out_key]

                        # 4) Parse
                        dets = parse_hailo_nms_by_class(nms_out, score_th=score_th, in_w=in_w, in_h=in_h, batch_idx=0)
                        if len(dets) > max_dets_total:
                            dets = dets[:max_dets_total]

                        # 5) Split + map
                
                        cyclists: List[Tuple[Tuple[int, int, int, int], float]] = []
                        numbers: List[Tuple[Tuple[int, int, int, int], float]] = []
                        for d in dets:
                            box = map_det_to_frame(d, fw, fh, in_w, in_h, r, pad_x, pad_y)
                            if d.class_id == class_cyclist:
                                cyclists.append((box, d.score))
                            elif d.class_id == class_number:
                                numbers.append((box, d.score))

                        # 6) Events (cross + association + OCR + uploader)
                        events_sent = 0
                        used_numbers = set()  # Track números ya procesados
                        
                        # FASE 1: Procesar ciclistas que cruzaron
                        for (cbox, cscore) in cyclists:
                            x1, y1, x2, y2 = cbox
                            bw = max(0, x2 - x1)
                            bh = max(0, y2 - y1)
                            if bw <= 0 or bh <= 0:
                                continue
                            cx, cy_bottom = bbox_bottom_center(x1, y1, x2, y2)
                            dist_to_line, crossed_now = gate.update(cx, cy_bottom, bw, bh)
                            if not crossed_now:
                                continue

                            nb = pick_number_for_cyclist(
                                cyclist_box=cbox,
                                number_dets=numbers,
                                inside_only=CFG["cross"]["number_inside_only"],
                                iou_fallback=CFG["cross"]["number_iou_fallback"],
                                min_iou=CFG["cross"]["min_number_iou"],
                            )

                            number_str = ""
                            plate_color = ""
                           
                            # OCR 
                            if (nb is not None) and (ocr is not None):
                                nx1, ny1, nx2, ny2 = nb
                                nx1, ny1, nx2, ny2 = clamp_box(nx1, ny1, nx2, ny2, fw, fh)
                                crop_rgb = frame_rgb[ny1:ny2, nx1:nx2]
                                if crop_rgb.size > 0:
                                    number_str, digits, _ , plate_color = ocr.read_number(crop_rgb, bgr = False)
                                used_numbers.add(nb)  # Marcar como usado
                                
                            ts_iso = datetime.now().astimezone().isoformat()
                            meta = {
                                "ts_utc": ts_iso,
                                "line_y": line_y,
                                "cyclist_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(cscore)},
                                "number_box": None if nb is None else {"x1": int(nb[0]), "y1": int(nb[1]), "x2": int(nb[2]), "y2": int(nb[3])},
                                "ocr": {"number": number_str, "plate_color": plate_color},
                                "dist_to_line_px": int(dist_to_line),
                                "detection_mode": "cyclist",
                            }

                            # Uploader
                            try:
                                write_event(number_str=number_str, ts_iso=ts_iso, frame_rgb=frame_rgb, meta=meta)
                            except Exception:
                                LOGGER.exception("Failed to write event")
                            events_sent += 1
                            LOGGER.info(
                                "Event [CYCLIST]: ts=%s cyclist_score=%.2f number='%s' plate=%s",
                                ts_iso, cscore, number_str, plate_color
                            )
                        
                        # FASE 2: Procesar números huérfanos que cruzaron
                        if number_gate is not None:
                            for (nbox, nscore) in numbers:
                                # Skip si ya fue procesado con un ciclista
                                if nbox in used_numbers:
                                    continue
                                
                                nx1, ny1, nx2, ny2 = nbox
                                nw = max(0, nx2 - nx1)
                                nh = max(0, ny2 - ny1)
                                if nw <= 0 or nh <= 0:
                                    continue
                                
                                ncx, ncy_bottom = bbox_bottom_center(nx1, ny1, nx2, ny2)
                                dist_to_line, crossed_now = number_gate.update(ncx, ncy_bottom, nw, nh)
                                
                                if not crossed_now:
                                    continue
                                
                                # Número cruzó sin ciclista - hacer OCR
                                number_str = ""
                                plate_color = ""
                                
                                if ocr is not None:
                                    nx1_c, ny1_c, nx2_c, ny2_c = clamp_box(nx1, ny1, nx2, ny2, fw, fh)
                                    crop_rgb = frame_rgb[ny1_c:ny2_c, nx1_c:nx2_c]
                                    if crop_rgb.size > 0:
                                        number_str, digits, _, plate_color = ocr.read_number(crop_rgb, bgr=False)
                                
                                ts_iso = datetime.now().astimezone().isoformat()
                                meta = {
                                    "ts_utc": ts_iso,
                                    "line_y": line_y,
                                    "cyclist_box": None,
                                    "number_box": {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2, "score": float(nscore)},
                                    "ocr": {"number": number_str, "plate_color": plate_color},
                                    "dist_to_line_px": int(dist_to_line),
                                    "detection_mode": "orphan",
                                }
                                
                                # Uploader
                                try:
                                    write_event(number_str=number_str, ts_iso=ts_iso, frame_rgb=frame_rgb, meta=meta)
                                except Exception:
                                    LOGGER.exception("Failed to write event (orphan)")
                                
                                events_sent += 1
                                LOGGER.info(
                                    "Event [ORPHAN]: ts=%s number='%s' plate=%s score=%.2f",
                                    ts_iso, number_str, plate_color, nscore
                                )

        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user.")
        finally:
            try:
                picam2.stop()
                writer_close()
            except Exception:
                pass
            LOGGER.info("Camera stopped.")


if __name__ == "__main__":
    main()
