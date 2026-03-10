#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import yaml

from picamera2 import Picamera2
from libcamera import Transform

import hailo_platform as hpf


# =========================
# Config
# =========================
DEFAULT_CONFIG = {
    "hef_path": "../models/vueltaalpartido_v1/raid_yolo.hef",
    "output_key": "yolov8n/yolov8_nms_postprocess",

    "main_size": [4656, 3496],
    "lores_size": [864, 480],
    "window_name": "lores_debug",
    "draw_debug": True,

    "camera_transform": {"vflip": True, "hflip": True},
    "camera_controls": {"AfMode": 2, "AfTrigger": 0, "AfSpeed": 1, "ExposureTime": 0, "AnalogueGain": 10},

    "inference": {
        "score_th": 0.05,
        "max_draw": 50,

        # For MODEL INPUT (not display):
        # - "RGB": keep camera frame as-is
        # - "BGR": swap channels before sending to model
        "model_channels": "RGB",

        # For cv2.imshow:
        # - "RGB2BGR": if lores is RGB (typical), convert to BGR for OpenCV display
        # - "NONE": show raw
        # - "BGR2RGB": if source is BGR and you want to display as RGB (rare)
        "display_convert": "RGB2BGR",

        # IMPORTANT: Many HEFs expect quantized uint8 input (0..255)
        "input_quantized_uint8": True,

        # Capture only lores for max FPS (recommended for debug)
        "capture_main": False,

        # Log timings every N frames
        "timing_every": 30,
    },
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
    cfg["draw_debug"] = bool(cfg.get("draw_debug", True))

    inf = cfg.get("inference", {})
    cfg["inference"] = {
        "score_th": float(inf.get("score_th", 0.05)),
        "max_draw": int(inf.get("max_draw", 50)),
        "model_channels": str(inf.get("model_channels", "RGB")).upper(),
        "display_convert": str(inf.get("display_convert", "RGB2BGR")).upper(),
        "input_quantized_uint8": bool(inf.get("input_quantized_uint8", True)),
        "capture_main": bool(inf.get("capture_main", False)),
        "timing_every": int(inf.get("timing_every", 30)),
    }

    if cfg["inference"]["model_channels"] not in ("RGB", "BGR"):
        cfg["inference"]["model_channels"] = "RGB"

    if cfg["inference"]["display_convert"] not in ("RGB2BGR", "BGR2RGB", "NONE"):
        cfg["inference"]["display_convert"] = "RGB2BGR"

    cfg["inference"]["timing_every"] = max(1, cfg["inference"]["timing_every"])
    return cfg


CFG = load_config()


# =========================
# Data structures
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
# Utilities (drawing)
# =========================
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def draw_box_pretty(img_rgb, x1, y1, x2, y2, text, color=(0, 255, 0)):
    """
    Draws a nicer bounding box + label on an RGB image.
    OpenCV drawing works fine on RGB, just remember to convert before imshow.
    """
    h, w = img_rgb.shape[:2]
    x1 = _clamp(int(x1), 0, w - 1)
    y1 = _clamp(int(y1), 0, h - 1)
    x2 = _clamp(int(x2), 0, w - 1)
    y2 = _clamp(int(y2), 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return

    th = 2 if max(h, w) < 800 else 3

    # Outer black border (shadow)
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 0), th + 2)
    # Inner colored box
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, th)

    # Corner accents
    corner = max(10, int(0.06 * min(x2 - x1, y2 - y1)))
    corner = min(corner, 30)
    # top-left
    cv2.line(img_rgb, (x1, y1), (x1 + corner, y1), color, th)
    cv2.line(img_rgb, (x1, y1), (x1, y1 + corner), color, th)
    # top-right
    cv2.line(img_rgb, (x2, y1), (x2 - corner, y1), color, th)
    cv2.line(img_rgb, (x2, y1), (x2, y1 + corner), color, th)
    # bottom-left
    cv2.line(img_rgb, (x1, y2), (x1 + corner, y2), color, th)
    cv2.line(img_rgb, (x1, y2), (x1, y2 - corner), color, th)
    # bottom-right
    cv2.line(img_rgb, (x2, y2), (x2 - corner, y2), color, th)
    cv2.line(img_rgb, (x2, y2), (x2, y2 - corner), color, th)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    pad = 5
    (tw, th_text), baseline = cv2.getTextSize(text, font, font_scale, 2)
    y_text = max(0, y1 - (th_text + baseline + 2 * pad))
    x_text = x1

    # Background and colored strip
    cv2.rectangle(
        img_rgb,
        (x_text, y_text),
        (x_text + tw + 2 * pad, y_text + th_text + baseline + 2 * pad),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        img_rgb,
        (x_text, y_text),
        (x_text + tw + 2 * pad, y_text + th_text + baseline + 2 * pad),
        color,
        -1,
    )

    # Text with outline
    cv2.putText(img_rgb, text, (x_text + pad, y_text + th_text + pad),
                font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_rgb, text, (x_text + pad, y_text + th_text + pad),
                font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def to_imshow(frame_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "NONE":
        return frame_rgb
    if mode == "RGB2BGR":
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if mode == "BGR2RGB":
        return cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    return frame_rgb


# =========================
# Preprocess
# =========================
def preprocess_uint8(lores_rgb: np.ndarray, in_w: int, in_h: int, model_channels: str) -> np.ndarray:
    """
    Returns UINT8 HWC in [0..255] (quantized input).
    Picamera2 RGB888 typically returns RGB order.
    """
    img = lores_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)


def preprocess_float32(lores_rgb: np.ndarray, in_w: int, in_h: int, model_channels: str) -> np.ndarray:
    """
    Returns FLOAT32 HWC in [0..1].
    """
    img = lores_rgb
    if model_channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0)


# =========================
# Output parsing
# =========================
def parse_hailo_nms_by_class(nms_out, score_th: float, in_w: int, in_h: int, batch_idx: int = 0) -> List[Det]:
    """
    nms_out: list[batch] of list[classes] of ndarray(N,5)
    row: [y_min, x_min, y_max, x_max, score]
    Coordinates may be normalized (0..1) OR pixels (0..in_w/in_h).
    """
    if not isinstance(nms_out, list) or len(nms_out) == 0:
        return []

    per_class = nms_out[batch_idx]
    dets: List[Det] = []

    # Probe coordinate scale using first non-empty class
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
                y1 *= in_h; y2 *= in_h
                x1 *= in_w; x2 *= in_w

            dets.append(Det(
                class_id=int(class_id),
                score=s,
                ymin=y1, xmin=x1, ymax=y2, xmax=x2
            ))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def map_det_to_lores(det: Det, lores_w: int, lores_h: int, in_w: int, in_h: int) -> Tuple[int, int, int, int]:
    """
    We used direct resize lores -> (in_w,in_h), so map back with scale.
    """
    sx = lores_w / float(in_w)
    sy = lores_h / float(in_h)

    x1 = int(det.xmin * sx)
    y1 = int(det.ymin * sy)
    x2 = int(det.xmax * sx)
    y2 = int(det.ymax * sy)

    x1 = max(0, min(lores_w - 1, x1))
    y1 = max(0, min(lores_h - 1, y1))
    x2 = max(0, min(lores_w - 1, x2))
    y2 = max(0, min(lores_h - 1, y2))

    return x1, y1, x2, y2


# =========================
# Main
# =========================
def main():
    hef_path = CFG["hef_path"]
    out_key_cfg = CFG["output_key"]

    score_th = CFG["inference"]["score_th"]
    max_draw = CFG["inference"]["max_draw"]
    model_channels = CFG["inference"]["model_channels"]
    display_convert = CFG["inference"]["display_convert"]

    input_quantized_uint8 = CFG["inference"]["input_quantized_uint8"]
    capture_main = CFG["inference"]["capture_main"]
    timing_every = CFG["inference"]["timing_every"]

    # -------------------------
    # Camera (Picamera2)
    # -------------------------
    picam2 = Picamera2()
    transform = Transform(
        vflip=CFG["camera_transform"]["vflip"],
        hflip=CFG["camera_transform"]["hflip"],
    )
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

    print("[INFO] IMX519 camera started.")

    lw, lh = CFG["lores_size"]

    # -------------------------
    # Hailo init
    # -------------------------
    hef = hpf.HEF(hef_path)
    input_infos = hef.get_input_vstream_infos()
    if len(input_infos) != 1:
        raise RuntimeError(f"Expected 1 input, got {len(input_infos)}")

    in_info = input_infos[0]
    in_h, in_w, in_c = in_info.shape
    if in_c != 3:
        raise RuntimeError(f"Expected 3 channels, got {in_c}")

    output_infos = hef.get_output_vstream_infos()
    out_names = [o.name for o in output_infos]

    out_key = out_key_cfg
    if out_key not in out_names:
        out_key = out_names[0]
        print(f"[WARN] output_key '{out_key_cfg}' not found. Using '{out_key}'.")

    print(f"[INFO] HEF loaded: {hef_path}")
    print(f"[INFO] Input: {in_info.name} shape={in_info.shape}")
    print("[INFO] Outputs:")
    for o in output_infos:
        print(f"  - {o.name} shape={o.shape} nms_shape={getattr(o, 'nms_shape', None)}")

    # display mode cycling
    modes = ["RGB2BGR", "NONE", "BGR2RGB"]
    mode_idx = modes.index(display_convert) if display_convert in modes else 0

    print("[INFO] Hotkeys: ESC quit | 'c' cycle display conversion")
    print(f"[INFO] model_channels={model_channels} input_quantized_uint8={input_quantized_uint8} capture_main={capture_main}")

    # -------------------------
    # Configure device once
    # -------------------------
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

        frames = 0
        fps = 0.0
        t_prev = time.perf_counter()

        last_log = 0

        with network_group.activate(ng_params):
            with hpf.InferVStreams(network_group, in_params, out_params) as pipe:
                while True:
                    t0 = time.perf_counter()

                    # Capture
                    if capture_main:
                        arrs, _ = picam2.capture_arrays(["lores", "main"])
                        lores = arrs[0]
                    else:
                        arrs, _ = picam2.capture_arrays(["lores"])
                        lores = arrs[0]

                    t1 = time.perf_counter()

                    # Preprocess
                    if input_quantized_uint8:
                        x = preprocess_uint8(lores, in_w, in_h, model_channels=model_channels)
                    else:
                        x = preprocess_float32(lores, in_w, in_h, model_channels=model_channels)

                    input_dict = {in_info.name: np.expand_dims(x, axis=0)}
                    t2 = time.perf_counter()

                    # Inference
                    infer_results = pipe.infer(input_dict)
                    nms_out = infer_results[out_key]
                    t3 = time.perf_counter()

                    dets = parse_hailo_nms_by_class(nms_out, score_th=score_th, in_w=in_w, in_h=in_h, batch_idx=0)

                    # Visualization
                    if CFG["draw_debug"]:
                        vis = lores.copy()  # RGB for drawing
                        for d in dets[:max_draw]:
                            x1, y1, x2, y2 = map_det_to_lores(d, lw, lh, in_w, in_h)
                            if x2 < x1: x1, x2 = x2, x1
                            if y2 < y1: y1, y2 = y2, y1

                            txt = f"ID {d.class_id}  {d.score:.2f}"
                            draw_box_pretty(vis, x1, y1, x2, y2, txt, color=(0, 255, 0))

                        # FPS (smoothed)
                        t_now = time.perf_counter()
                        dt = max(1e-6, t_now - t_prev)
                        inst_fps = 1.0 / dt
                        fps = 0.9 * fps + 0.1 * inst_fps
                        t_prev = t_now

                        cv2.putText(
                            vis,
                            f"FPS: {fps:.1f} | mode: {modes[mode_idx]} | score_th: {score_th}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )

                        cv2.imshow(CFG["window_name"], to_imshow(vis, modes[mode_idx]))

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC
                            break
                        if key == ord("c"):
                            mode_idx = (mode_idx + 1) % len(modes)

                    t4 = time.perf_counter()

                    frames += 1

                    # Timings
                    if frames - last_log >= timing_every:
                        cap_ms = (t1 - t0) * 1000.0
                        pre_ms = (t2 - t1) * 1000.0
                        inf_ms = (t3 - t2) * 1000.0
                        vis_ms = (t4 - t3) * 1000.0
                        print(f"[TIMING] cap={cap_ms:.1f}ms pre={pre_ms:.1f}ms inf={inf_ms:.1f}ms vis={vis_ms:.1f}ms dets={len(dets)}")
                        last_log = frames

    # Cleanup
    try:
        picam2.stop()
    except Exception:
        pass
    if CFG["draw_debug"]:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
