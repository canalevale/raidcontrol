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
def get_logger(name: str = "BikeDetectorDBG") -> logging.Logger:
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


LOGGER = get_logger("BikeDetectorDBG")


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
    },

    # Logging
    "logging": {
        "level": "INFO",
        "timing_every_frames": 60,
    },

    # Debug UI
    "debug_ui": {
        "enabled": True,
        "window_name": "RaidControl DEBUG",
        "waitkey_ms": 1,
        "show_hud": True,
        "show_boxes": True,
        "show_numbers": True,
        "show_cross_state": True,
        "draw_thickness": 2,
        "font_scale": 1,
        "save_dir": "debug_snaps",
        "resize_preview": [2328, 1748],  # set null/empty to keep native
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
    }

    logcfg = cfg.get("logging", {})
    cfg["logging"] = {
        "level": str(logcfg.get("level", "INFO")).upper(),
        "timing_every_frames": int(logcfg.get("timing_every_frames", 60)),
    }
    cfg["logging"]["timing_every_frames"] = max(1, cfg["logging"]["timing_every_frames"])

    dbg = cfg.get("debug_ui", {})
    dflt = DEFAULT_CONFIG["debug_ui"]
    cfg["debug_ui"] = {
        "enabled": bool(dbg.get("enabled", dflt["enabled"])),
        "window_name": str(dbg.get("window_name", dflt["window_name"])),
        "waitkey_ms": int(dbg.get("waitkey_ms", dflt["waitkey_ms"])),
        "show_hud": bool(dbg.get("show_hud", dflt["show_hud"])),
        "show_boxes": bool(dbg.get("show_boxes", dflt["show_boxes"])),
        "show_numbers": bool(dbg.get("show_numbers", dflt["show_numbers"])),
        "show_cross_state": bool(dbg.get("show_cross_state", dflt["show_cross_state"])),
        "draw_thickness": int(dbg.get("draw_thickness", dflt["draw_thickness"])),
        "font_scale": float(dbg.get("font_scale", dflt["font_scale"])),
        "save_dir": str(dbg.get("save_dir", dflt["save_dir"])),
        "resize_preview": dbg.get("resize_preview", dflt["resize_preview"]),
    }
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
def preprocess_uint8(frame_rgb: np.ndarray, in_w: int, in_h: int, model_channels: str) -> np.ndarray:
    img = frame_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)


def preprocess_float32(frame_rgb: np.ndarray, in_w: int, in_h: int, model_channels: str) -> np.ndarray:
    img = frame_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0)


def parse_hailo_nms_by_class(nms_out, score_th: float, in_w: int, in_h: int, batch_idx: int = 0) -> List[Det]:
    if not isinstance(nms_out, list) or len(nms_out) == 0:
        return []

    per_class = nms_out[batch_idx]
    dets: List[Det] = []

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

            y1 = float(y1); x1 = float(x1); y2 = float(y2); x2 = float(x2)

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


def map_det_to_frame(det: Det, frame_w: int, frame_h: int, in_w: int, in_h: int) -> Tuple[int, int, int, int]:
    sx = frame_w / float(in_w)
    sy = frame_h / float(in_h)

    x1 = int(det.xmin * sx)
    y1 = int(det.ymin * sy)
    x2 = int(det.xmax * sx)
    y2 = int(det.ymax * sy)

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))

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
# Debug draw helpers
# =========================
def draw_box(img_bgr, box, label, thickness=2, font_scale=0.6):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    if label:
        y = max(20, y1 - 8)
        cv2.putText(img_bgr, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)


def draw_hud(img_bgr, lines: List[str], font_scale=0.6):
    y = 22
    for s in lines:
        cv2.putText(img_bgr, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += int(22 * (font_scale / 0.6))


# =========================
# OCR
# =========================
ocr = None
if CFG["ocr"]["enabled"]:
    ocr = DigitReaderCNNONNX(onnx_path=CFG["ocr"]["model_ocr_path"], yaml_path="config.yaml")


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
    print("Camera controls applied:", CFG["camera_controls"])
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

    timing_every_frames = CFG["logging"]["timing_every_frames"]
    ts = TimingStats()
    t_window0 = time.perf_counter()
    frames_window = 0
    fps_win = 0.0
    last_perf_ms = {}

    # Debug UI toggles (runtime)
    dbg = CFG["debug_ui"]
    win = dbg["window_name"]
    os.makedirs(dbg["save_dir"], exist_ok=True)

    paused = False
    enable_ocr = (ocr is not None)
    enable_uploader = True
    show_detail = True

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

    LOGGER.info("HEF loaded: %s", CFG["hef_path"])
    LOGGER.info("Input: %s shape=%s", in_info.name, in_info.shape)
    LOGGER.info("Output key: %s", out_key)
    LOGGER.info("Line Y (main): %d px", line_y)

    if dbg["enabled"]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        LOGGER.info("DEBUG window enabled. Keys: q/ESC quit | s snapshot | p pause | o OCR | u uploader | d detail")

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
                        # Pause handling
                        if paused:
                            k = cv2.waitKey(30) & 0xFF
                            if k in (ord('q'), 27):
                                break
                            if k == ord('p'):
                                paused = False
                            continue

                        # 1) Capture
                        t_cap0 = time.perf_counter()
                        frame_rgb = picam2.capture_array("main")  # RGB
                        t_cap1 = time.perf_counter()

                        # 2) Preprocess
                        t_pre0 = time.perf_counter()
                        if input_quantized_uint8:
                            x = preprocess_uint8(frame_rgb, in_w, in_h, model_channels=model_channels)
                        else:
                            x = preprocess_float32(frame_rgb, in_w, in_h, model_channels=model_channels)
                        input_dict = {in_info.name: np.expand_dims(x, axis=0)}
                        t_pre1 = time.perf_counter()

                        # 3) Inference
                        t_inf0 = time.perf_counter()
                        infer_results = pipe.infer(input_dict)
                        nms_out = infer_results[out_key]
                        t_inf1 = time.perf_counter()

                        # 4) Parse
                        t_par0 = time.perf_counter()
                        dets = parse_hailo_nms_by_class(nms_out, score_th=score_th, in_w=in_w, in_h=in_h, batch_idx=0)
                        if len(dets) > max_dets_total:
                            dets = dets[:max_dets_total]
                        t_par1 = time.perf_counter()

                        # 5) Split + map
                        t_map0 = time.perf_counter()
                        cyclists: List[Tuple[Tuple[int, int, int, int], float]] = []
                        numbers: List[Tuple[Tuple[int, int, int, int], float]] = []
                        for d in dets:
                            box = map_det_to_frame(d, fw, fh, in_w, in_h)
                            if d.class_id == class_cyclist:
                                cyclists.append((box, d.score))
                            elif d.class_id == class_number:
                                numbers.append((box, d.score))
                        t_map1 = time.perf_counter()

                        # 6) Events (cross + association + OCR + uploader)
                        t_evt0 = time.perf_counter()
                        events_sent = 0
                        last_event_text = ""

                        for (cbox, cscore) in cyclists:
                            x1, y1, x2, y2 = cbox
                            bw = max(0, x2 - x1)
                            bh = max(0, y2 - y1)
                            if bw <= 0 or bh <= 0:
                                continue

                            cx, cy_bottom = bbox_bottom_center(x1, y1, x2, y2)
                            dist_to_line, crossed_now = gate.update(cx, cy_bottom, bw, bh)

                            nb = None
                            
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
                            t_ocr0 = time.perf_counter()
                            if enable_ocr and (nb is not None) and (ocr is not None):
                                nx1, ny1, nx2, ny2 = nb
                                nx1, ny1, nx2, ny2 = clamp_box(nx1, ny1, nx2, ny2, fw, fh)
                                crop_rgb = frame_rgb[ny1:ny2, nx1:nx2]
                                if crop_rgb.size > 0:
                                    number_str, digits, _ , plate_color = ocr.read_number(crop_rgb, bgr = False)
                            t_ocr1 = time.perf_counter()
                            ts.add("ocr", t_ocr1 - t_ocr0)

                            if crossed_now:
                                ts_iso = datetime.now().astimezone().isoformat()
                                if (number_str is None) or (str(number_str).strip() == ""):
                                    number_str = "NA"
                                meta = {
                                    "ts_utc": ts_iso,
                                    "line_y": line_y,
                                    "cyclist_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(cscore)},
                                    "number_box": None if nb is None else {"x1": int(nb[0]), "y1": int(nb[1]), "x2": int(nb[2]), "y2": int(nb[3])},
                                    "ocr": {"number": number_str, "plate_color": plate_color},
                                    "dist_to_line_px": int(dist_to_line),
                                }

                                # Uploader
                                t_up0 = time.perf_counter()
                                if enable_uploader:
                                    write_event(number_str=number_str, ts_iso=ts_iso, frame_rgb=frame_rgb, meta=meta)
                                t_up1 = time.perf_counter()
                                ts.add("uploader", t_up1 - t_up0)

                                events_sent += 1
                                last_event_text = f"Event: num='{number_str}' plate={plate_color}"
                                LOGGER.info(
                                    "Event: ts=%s cyclist_score=%.2f number='%s' plate=%s",
                                    ts_iso, cscore, number_str, plate_color
                                )

                        t_evt1 = time.perf_counter()

                        # Record per-frame timings
                        ts.add("capture", t_cap1 - t_cap0)
                        ts.add("preprocess", t_pre1 - t_pre0)
                        ts.add("infer", t_inf1 - t_inf0)
                        ts.add("parse", t_par1 - t_par0)
                        ts.add("map", t_map1 - t_map0)
                        ts.add("events_total", t_evt1 - t_evt0)

                        # Window report
                        frames_window += 1
                        if frames_window >= timing_every_frames:
                            t_window1 = time.perf_counter()
                            elapsed = max(1e-6, t_window1 - t_window0)
                            fps_win = frames_window / elapsed
                            last_perf_ms = ts.snapshot_ms()
                            LOGGER.info(
                                "Perf: fps=%.1f | cap=%.1fms pre=%.1fms inf=%.1fms parse=%.1fms map=%.1fms evt=%.1fms ocr=%.1fms up=%.1fms | dets=%d cyclists=%d nums=%d events=%d",
                                fps_win,
                                last_perf_ms.get("capture", 0.0),
                                last_perf_ms.get("preprocess", 0.0),
                                last_perf_ms.get("infer", 0.0),
                                last_perf_ms.get("parse", 0.0),
                                last_perf_ms.get("map", 0.0),
                                last_perf_ms.get("events_total", 0.0),
                                last_perf_ms.get("ocr", 0.0),
                                last_perf_ms.get("uploader", 0.0),
                                len(dets),
                                len(cyclists),
                                len(numbers),
                                events_sent,
                            )
                            t_window0 = t_window1
                            frames_window = 0
                            ts.reset()

                        # ----------------------------
                        # DEBUG UI RENDER
                        # ----------------------------
                        if dbg["enabled"]:
                            # OpenCV shows BGR, convert for display
                            disp = frame_rgb

                            # Draw finish line
                            cv2.line(disp, (0, line_y), (fw - 1, line_y), (0, 0, 255), dbg["draw_thickness"])

                            if dbg["show_boxes"]:
                                # cyclists
                                for (cbox, cscore) in cyclists:
                                    x1, y1, x2, y2 = cbox
                                    label = f"cyclist {cscore:.2f}"
                                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), dbg["draw_thickness"])
                                    cv2.putText(disp, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                                dbg["font_scale"], (0, 255, 0), 2, cv2.LINE_AA)

                                    if dbg["show_cross_state"]:
                                        cx, cyb = bbox_bottom_center(x1, y1, x2, y2)
                                        dist = int(cyb - line_y)
                                        txt = f"dist={dist}px"
                                        cv2.circle(disp, (cx, cyb), 4, (255, 255, 0), -1)
                                        cv2.putText(disp, txt, (x1, min(fh - 10, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                                                    dbg["font_scale"], (255, 255, 0), 2, cv2.LINE_AA)

                                # number boxes
                                if dbg["show_numbers"]:
                                    for (nb, ns) in numbers:
                                        nx1, ny1, nx2, ny2 = nb
                
                                        cv2.rectangle(disp, (nx1, ny1), (nx2, ny2), (255, 0, 0), dbg["draw_thickness"])
                                        cv2.putText(disp, f"num,{ns:.2f}", (nx1, max(20, ny1 - 8)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, dbg["font_scale"], (255, 0, 0), 2, cv2.LINE_AA)

                            if dbg["show_hud"]:
                                ms = last_perf_ms or {}
                                hud = [
                                    f"FPS(win)={fps_win:.1f} | dets={len(dets)} cy={len(cyclists)} num={len(numbers)}",
                                    f"cap={ms.get('capture',0):.1f}ms pre={ms.get('preprocess',0):.1f}ms inf={ms.get('infer',0):.1f}ms parse={ms.get('parse',0):.1f}ms map={ms.get('map',0):.1f}ms",
                                    f"evt={ms.get('events_total',0):.1f}ms ocr={ms.get('ocr',0):.1f}ms up={ms.get('uploader',0):.1f}ms",
                                    f"OCR={'ON' if enable_ocr else 'OFF'} | Uploader={'ON' if enable_uploader else 'OFF'} | Detail={'ON' if show_detail else 'OFF'}",
                                ]
                                if show_detail and last_event_text:
                                    hud.append(last_event_text)
                                draw_hud(disp, hud, font_scale=dbg["font_scale"])

                            # Resize preview if configured
                            rp = dbg.get("resize_preview")
                            if rp and isinstance(rp, (list, tuple)) and len(rp) == 2:
                                pw, ph = int(rp[0]), int(rp[1])
                                if pw > 0 and ph > 0:
                                    disp = cv2.resize(disp, (pw, ph), interpolation=cv2.INTER_AREA)

                            cv2.imshow(win, disp)
                            k = cv2.waitKey(dbg["waitkey_ms"]) & 0xFF

                            if k in (ord('q'), 27):
                                break
                            elif k == ord('s'):
                                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                outp = os.path.join(dbg["save_dir"], f"snap_{ts_name}.jpg")
                                cv2.imwrite(outp, disp)
                                LOGGER.info("Saved snapshot: %s", outp)
                            elif k == ord('p'):
                                paused = True
                                LOGGER.info("Paused. Press 'p' to resume.")
                            elif k == ord('o'):
                                enable_ocr = not enable_ocr
                                LOGGER.info("OCR toggled: %s", "ON" if enable_ocr else "OFF")
                            elif k == ord('u'):
                                enable_uploader = not enable_uploader
                                LOGGER.info("Uploader toggled: %s", "ON" if enable_uploader else "OFF")
                            elif k == ord('d'):
                                show_detail = not show_detail
                                LOGGER.info("Detail toggled: %s", "ON" if show_detail else "OFF")

        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user.")
        finally:
            try:
                picam2.stop()
                cv2.destroyAllWindows()
                writer_close()
            except Exception:
                pass
            LOGGER.info("Camera stopped.")


if __name__ == "__main__":
    main()
