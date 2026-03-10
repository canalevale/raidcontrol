#!/usr/bin/env python3
import os
import time
from datetime import datetime

import cv2
import numpy as np

from picamera2 import Picamera2
from libcamera import Transform
import hailo_platform as hpf

# =========================
# Config mínima
# =========================
SAVE_DIR = "clips_test"
os.makedirs(SAVE_DIR, exist_ok=True)

HEF_PATH = "../models/vueltaalpartido_v1/raid_yolo.hef"
OUTPUT_KEY = "yolov8n/yolov8_nms_postprocess"

DETECT_CLASS_ID = 0
SCORE_TH = 0.4

LORES_SIZE = (864, 480)
MAIN_SIZE = (4656, 3496)

MIN_BOX_W = 40
MIN_BOX_H = 40

COOLDOWN_SEC = 0  # evita guardar 30 veces el mismo objeto

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

def parse_hailo_nms(nms_out, score_th, in_w, in_h):
    dets = []
    per_class = nms_out[0]

    for class_id, arr in enumerate(per_class):
        if arr is None:
            continue
        for y1, x1, y2, x2, s in arr:
            if s < score_th:
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1 *= in_w; x2 *= in_w
                y1 *= in_h; y2 *= in_h
            dets.append((class_id, s, x1, y1, x2, y2))
    return dets

# =========================
# Camera
# =========================
picam2 = Picamera2()
transform = Transform(vflip=True, hflip=True)

config = picam2.create_preview_configuration(
    main={"size": MAIN_SIZE, "format": "RGB888"},
    transform=transform,
)

picam2.configure(config)
picam2.set_controls({"AfMode": 0, "AfSpeed":1 ,"LensPosition": 4, "AeExposureMode": 1, "ExposureTime":0, "AnalogueGain":0})

picam2.start()
print("[INFO] Camera started")

# =========================
# Hailo init
# =========================
hef = hpf.HEF(HEF_PATH)
in_info = hef.get_input_vstream_infos()[0]
out_key = OUTPUT_KEY

with hpf.VDevice() as vdevice:
    cfg = hpf.ConfigureParams.create_from_hef(hef,interface=hpf.HailoStreamInterface.PCIe)
    ng = vdevice.configure(hef, cfg)[0]
    ng_params = ng.create_params()

    in_params = hpf.InputVStreamParams.make_from_network_group(
        ng, quantized=True, format_type=hpf.FormatType.UINT8
    )
    out_params = hpf.OutputVStreamParams.make_from_network_group(
        ng, quantized=False, format_type=hpf.FormatType.FLOAT32
    )

    last_save = 0

    with ng.activate(ng_params):
        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            print("[INFO] Running fast detection loop")
            try:
                while True:
                    # ---- CAPTURE LORES ONLY ----
                    lores = picam2.capture_array("main")

                    # ---- PREPROCESS ----
                    x = cv2.resize(lores, (in_info.shape[1], in_info.shape[0]))
                    x = np.expand_dims(x.astype(np.uint8), axis=0)

                    # ---- INFERENCE ----
                    res = pipe.infer({in_info.name: x})
                    dets = parse_hailo_nms(res[out_key], SCORE_TH,
                                           in_info.shape[1], in_info.shape[0])

                    # ---- CHECK DETECTIONS ----
                    for cls, score, x1, y1, x2, y2 in dets:
                        if cls != DETECT_CLASS_ID:
                            continue

                        now = time.time()
                        if now - last_save < COOLDOWN_SEC:
                            continue

                        # ---- CAPTURE FULL FRAME (ONCE) ----
            

                        # sx = MAIN_SIZE[0] / LORES_SIZE[0]
                        # sy = MAIN_SIZE[1] / LORES_SIZE[1]

                        # x1f = int(x1 * sx)
                        # y1f = int(y1 * sy)
                        # x2f = int(x2 * sx)
                        # y2f = int(y2 * sy)

                        # x1f, y1f, x2f, y2f = clamp_box(
                        #     x1f, y1f, x2f, y2f, *MAIN_SIZE
                        # )

                        # if x2f - x1f < MIN_BOX_W or y2f - y1f < MIN_BOX_H:
                        #     continue


                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fname = f"det_{cls}_{score:.2f}_{ts}.jpg"
                        cv2.imwrite(
                            os.path.join(SAVE_DIR, fname),
                            lores
                        )

                        print(f"[SAVE] {fname}")
                        last_save = now
                        break

            except KeyboardInterrupt:
                print("\n[INFO] Stopped")

picam2.stop()
