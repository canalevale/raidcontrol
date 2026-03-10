import os
import yaml
import cv2

from picamera2 import Picamera2
from libcamera import Transform, controls


DEFAULT_CONFIG = {
    "main_size": (1280, 720),
    "lores_size": (640, 360),
    "camera_transform": {"vflip": False, "hflip": False},
    "camera_controls": {
        # Keep your style: AF params live here
        "AfMode": 2,         # 1=Auto, 2=Continuous, 3=Manual
        "AfTrigger": 0,      # 0=Idle, 1=Start (only used with AfMode=Auto)
        "AfSpeed": 1,        # 0=Normal, 1=Fast (only for AF modes)
        "ExposureTime": 0,   # 0=auto exposure
        "AnalogueGain": 20,
        # "LensPosition": 8.5 # optional default manual focus value (diopters)
    },
}


def load_config(path="config2.yaml"):
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

    ct = cfg.get("camera_transform", {}) or {}
    cfg["camera_transform"] = {
        "vflip": bool(ct.get("vflip", DEFAULT_CONFIG["camera_transform"]["vflip"])),
        "hflip": bool(ct.get("hflip", DEFAULT_CONFIG["camera_transform"]["hflip"])),
    }

    cfg["camera_controls"] = cfg.get("camera_controls", DEFAULT_CONFIG["camera_controls"]) or {}
    return cfg


def get_lens_position_range(picam2):
    """
    Try to read LensPosition range from picam2.camera_controls.
    Fallback to 0..32 diopters.
    """
    lens_min, lens_max, lens_step = 0.0, 32.0, 0.1
    try:
        cc = picam2.camera_controls
        if "LensPosition" in cc:
            lp = cc["LensPosition"]  # usually (min, max, step)
            lens_min, lens_max, lens_step = float(lp[0]), float(lp[1]), float(lp[2])
    except Exception:
        pass
    return lens_min, lens_max, lens_step


def apply_camera_controls(picam2, camera_controls: dict):
    """
    Apply controls from YAML, but:
    - If LensPosition is present -> FORCE AfMode=Manual (required).
    - Otherwise apply as-is.
    """
    to_apply = dict(camera_controls)

    if "LensPosition" in to_apply:
        to_apply["AfMode"] = controls.AfModeEnum.Manual
        print(f"[INFO] Manual focus default: LensPosition={to_apply['LensPosition']}")
    else:
        print(f"[INFO] Autofocus config: AfMode={to_apply.get('AfMode', 'default')}")

    picam2.set_controls(to_apply)


def run_focus_tuning_ui(picam2, cfg):
    """
    Manual-focus tuning window.
    - Forces AfMode=Manual while UI is running.
    - Slider controls LensPosition.
    - Shows LensPosition (and rough distance) on-screen.
    Keys:
      q = quit
    """
    lens_min, lens_max, lens_step = get_lens_position_range(picam2)
    print(f"[INFO] LensPosition range: min={lens_min} max={lens_max} step={lens_step}")

    SCALE = 100  # 0.01 diopter resolution
    tb_min = int(round(lens_min * SCALE))
    tb_max = int(round(lens_max * SCALE))

    cc = cfg["camera_controls"]
    current_lp = float(cc.get("LensPosition", (lens_min + lens_max) / 2.0))
    tb_init = int(round(current_lp * SCALE))
    tb_init = max(tb_min, min(tb_max, tb_init))

    # Force manual focus immediately
    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(current_lp)})
    except Exception as e:
        print(f"[WARN] Could not set initial manual focus: {e}")

    win = "Focus tuning (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_trackbar(val):
        nonlocal current_lp
        current_lp = val / SCALE
        try:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(current_lp)})
        except Exception as e:
            print(f"[WARN] set_controls failed: {e}")

    cv2.createTrackbar("LensPosition", win, tb_init, tb_max, on_trackbar)
    cv2.setTrackbarMin("LensPosition", win, tb_min)
    on_trackbar(tb_init)

    while True:
        # Picamera2 main stream is configured as RGB888 => this returns RGB order
        rgb = picam2.capture_array("main")  # RGB

        # OpenCV expects BGR for correct display, so convert RGB -> BGR

        dist_m = (1.0 / current_lp) if current_lp > 1e-6 else 0.0

        cv2.putText(
            rgb,
            f"LensPosition: {current_lp:.2f} D   (~{dist_m:.2f} m)",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )

        cv2.imshow(win, rgb)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    cfg = load_config("config2.yaml")

    picam2 = Picamera2()

    transform = Transform(
        vflip=cfg["camera_transform"]["vflip"],
        hflip=cfg["camera_transform"]["hflip"],
    )

    cam_config = picam2.create_preview_configuration(
        main={"size": cfg["main_size"], "format": "RGB888"},
        transform=transform,
    )

    picam2.configure(cam_config)
    picam2.start()

    try:
        # Apply YAML controls first (keeps your AfMode/AfTrigger/AfSpeed, etc.)
        try:
            apply_camera_controls(picam2, cfg["camera_controls"])
        except Exception as e:
            print(f"[WARN] Could not apply some camera controls: {e}")

        # Open tuning UI (forces manual focus while it runs)
        run_focus_tuning_ui(picam2, cfg)

    finally:
        picam2.stop()
        print("[INFO] Camera stopped.")


if __name__ == "__main__":
    main()
