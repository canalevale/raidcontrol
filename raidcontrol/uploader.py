
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import yaml
import requests
from dotenv import load_dotenv

import cv2
import numpy as np


# ============================================================
# Logging
# ============================================================
def get_logger(name: str = "uploader") -> logging.Logger:
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


LOG = get_logger("Uploader")


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # station
    station_id: str
    device_id: Optional[str]

    # local store
    db_path: str
    images_dir: str

    # backend (proposed)
    backend_base_url: str
    create_event_path: str            # POST /api/v1/stations/{station_id}/events
    upload_image_path_tmpl: str       # POST /api/v1/events/{event_id}/image
    device_api_key: Optional[str]

    # uploader runtime
    uploader_enabled: bool
    poll_interval_sec: float
    batch_size: int
    request_timeout_sec: float
    max_retries: int
    backoff_base_sec: float
    backoff_max_sec: float


def load_config(path: str) -> Config:
    # Cargar variables de entorno desde .env
    load_dotenv()
    
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    station = d.get("station", {})
    local_store = d.get("local_store", {})
    backend = d.get("backend", {})
    uploader = d.get("uploader", {})

    # Leer variables sensibles desde .env, con fallback a config.yaml (deprecated)
    api_base_url = os.getenv("API_BASE_URL") or backend.get("api_base_url", "http://127.0.0.1:8000")
    device_api_key = os.getenv("DEVICE_API_KEY") or backend.get("device_api_key", "")

    return Config(
        station_id=str(station.get("station_id", "unknown-station")),
        device_id=station.get("device_id"),

        db_path=str(local_store.get("db_path", "./local/uploader.db")),
        images_dir=str(local_store.get("images_dir", "./local/images")),

        backend_base_url=str(api_base_url),
        create_event_path=str(backend.get("create_event_path", "/api/v1/stations/{station_id}/events")),
        upload_image_path_tmpl=str(backend.get("upload_image_path_tmpl", "/api/v1/events/{event_id}/image")),
        device_api_key=str(device_api_key),

        uploader_enabled=bool(uploader.get("enabled", True)),
        poll_interval_sec=float(uploader.get("poll_interval_sec", 0.5)),
        batch_size=int(uploader.get("batch_size", 10)),
        request_timeout_sec=float(uploader.get("request_timeout_sec", 10.0)),
        max_retries=int(uploader.get("max_retries", 50)),
        backoff_base_sec=float(uploader.get("backoff_base_sec", 1.0)),
        backoff_max_sec=float(uploader.get("backoff_max_sec", 60.0)),
    )


# ============================================================
# SQLite schema (minimal columns as requested)
# ============================================================
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  local_uuid TEXT NOT NULL UNIQUE,

  station_id TEXT NOT NULL,
  device_id TEXT,

  detected_at TEXT NOT NULL,     -- ISO-8601 UTC (from your ts_iso)
  created_at TEXT NOT NULL,      -- ISO-8601 UTC (insert time)

  number_str TEXT NOT NULL,      -- detected number
  image_rel_path TEXT,           -- relative path inside images_dir (nullable)
  meta_json TEXT NOT NULL,       -- full meta dict as JSON

  status TEXT NOT NULL,          -- pending|uploading|sent|failed
  retries INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,

  remote_event_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_status_created ON events(status, created_at);
CREATE INDEX IF NOT EXISTS idx_events_station_detected ON events(station_id, detected_at);
CREATE INDEX IF NOT EXISTS idx_events_station_number ON events(station_id, number_str);
"""


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def db_connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_db(conn)
    return conn


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(_sanitize(obj), ensure_ascii=False)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ============================================================
# Writer
# ============================================================
_WRITER_STATE: Optional[Tuple[Config, sqlite3.Connection]] = None


def writer_init(cfg_path: str) -> None:
    """
    Call once in the main process at startup.
    Loads config, ensures DB exists, opens one sqlite connection.
    """
    global _WRITER_STATE
    if _WRITER_STATE is not None:
        return

    cfg = load_config(cfg_path)
    os.makedirs(cfg.images_dir, exist_ok=True)

    conn = db_connect(cfg.db_path)
    _WRITER_STATE = (cfg, conn)

    LOG.info(
        f"Writer initialized | station_id={cfg.station_id} db={cfg.db_path} images={cfg.images_dir}"
    )


def writer_close() -> None:
    global _WRITER_STATE
    if _WRITER_STATE is None:
        return
    cfg, conn = _WRITER_STATE
    conn.close()
    _WRITER_STATE = None
    LOG.info("Writer closed.")


def _save_frame_rgb(cfg: Config, local_uuid: str, frame_rgb: "np.ndarray") -> Optional[str]:
    if cv2 is None:
        LOG.error("cv2 not available, cannot save images.")
        return None

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dst_dir = os.path.join(cfg.images_dir, day)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, f"{local_uuid}.jpg")
    ok = cv2.imwrite(dst_path, frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        LOG.error("cv2.imwrite failed.")
        return None

    return os.path.relpath(dst_path, cfg.images_dir)


def write_event(
    number_str: str,
    ts_iso: str,
    frame_rgb: Optional["np.ndarray"],
    meta: Dict[str, Any],
) -> str:
    """
    Fast path called by the main detection pipeline.
    Requires writer_init() to have been called once.
    """
    global _WRITER_STATE
    if _WRITER_STATE is None:
        raise RuntimeError("Writer not initialized. Call writer_init(cfg_path) once at startup.")

    cfg, conn = _WRITER_STATE
    local_uuid = str(uuid.uuid4())

    image_rel_path = None
    if frame_rgb is not None:
        image_rel_path = _save_frame_rgb(cfg, local_uuid, frame_rgb)

    meta_json = json_dumps_safe(meta)
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO events (
          local_uuid,
          station_id, device_id,
          detected_at, created_at,
          number_str, image_rel_path, meta_json,
          status, retries, last_error, remote_event_id
        ) VALUES (
          ?,
          ?, ?,
          ?, ?,
          ?, ?, ?,
          'pending', 0, NULL, NULL
        )
        """,
        (
            local_uuid,
            cfg.station_id, cfg.device_id,
            ts_iso, now_iso,
            number_str, image_rel_path, meta_json,
        ),
    )
    conn.commit()
    return local_uuid


# ============================================================
# Uploader daemon side (separate process)
# ============================================================
def _fetch_pending(conn: sqlite3.Connection, limit: int):
    cur = conn.execute(
        """
        SELECT * FROM events
        WHERE status IN ('pending','failed')
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()


def _mark_uploading(conn: sqlite3.Connection, event_id: int) -> None:
    conn.execute("UPDATE events SET status='uploading' WHERE id=?", (event_id,))
    conn.commit()


def _mark_sent(conn: sqlite3.Connection, event_id: int) -> None:
    conn.execute("UPDATE events SET status='sent', last_error=NULL WHERE id=?", (event_id,))
    conn.commit()


def _mark_failed(conn: sqlite3.Connection, event_id: int, msg: str) -> None:
    conn.execute(
        """
        UPDATE events
        SET status='failed',
            retries=retries+1,
            last_error=?
        WHERE id=?
        """,
        (msg[:2000], event_id),
    )
    conn.commit()


def _set_remote_event_id(conn: sqlite3.Connection, event_id: int, remote_id: str) -> None:
    conn.execute("UPDATE events SET remote_event_id=? WHERE id=?", (remote_id, event_id))
    conn.commit()


def _backoff_sleep(cfg: Config, retries: int) -> None:
    if retries <= 0:
        return
    t = min(cfg.backoff_base_sec * (2 ** (retries - 1)), cfg.backoff_max_sec)
    time.sleep(t)


def _create_event_on_backend(cfg: Config, row: sqlite3.Row, session: requests.Session) -> str:
    base = cfg.backend_base_url.rstrip("/")
    url = base + cfg.create_event_path.format(station_id=row["station_id"])

    payload = {
        "station_id": row["station_id"],
        "device_id": row["device_id"],
        "detected_at": row["detected_at"],
        "local_uuid": row["local_uuid"],
        "number_str": row["number_str"],
        "meta": json.loads(row["meta_json"]),
    }

    r = session.post(url, json=payload, timeout=cfg.request_timeout_sec)
    if r.status_code >= 300:
        raise RuntimeError(f"create_event failed | status={r.status_code} body={r.text[:400]}")

    data = r.json()
    remote_id = data.get("event_id") or data.get("id")
    if not remote_id:
        raise RuntimeError(f"create_event missing event_id | body={data}")

    return str(remote_id)


def _upload_image(cfg: Config, remote_event_id: str, image_rel_path: str, session: requests.Session) -> None:
    base = cfg.backend_base_url.rstrip("/")
    url = base + cfg.upload_image_path_tmpl.format(event_id=remote_event_id)

    abs_path = os.path.join(cfg.images_dir, image_rel_path)
    if not os.path.exists(abs_path):
        raise RuntimeError(f"image not found | path={abs_path}")

    with open(abs_path, "rb") as f:
        files = {"file": (os.path.basename(abs_path), f, "image/jpeg")}
        r = session.post(url, files=files, timeout=cfg.request_timeout_sec)

    if r.status_code >= 300:
        raise RuntimeError(f"upload_image failed | status={r.status_code} body={r.text[:400]}")


def run_uploader(cfg_path: str) -> None:
    """
    Run the uploader loop as a standalone process:
      python3 uploader.py config.yaml
    """
    cfg = load_config(cfg_path)
    if not cfg.uploader_enabled:
        LOG.info("Uploader disabled by config. Exiting.")
        return

    os.makedirs(cfg.images_dir, exist_ok=True)
    conn = db_connect(cfg.db_path)
    session = requests.Session()
    session.headers["X-Device-Key"] = cfg.device_api_key

    LOG.info(
        f"Uploader started | station_id={cfg.station_id} db={cfg.db_path} api={cfg.backend_base_url}"
    )

    while True:
        batch = _fetch_pending(conn, cfg.batch_size)
        if not batch:
            time.sleep(cfg.poll_interval_sec)
            continue

        for row in batch:
            event_id = int(row["id"])
            retries = int(row["retries"])
            if retries >= cfg.max_retries:
                continue

            try:
                _backoff_sleep(cfg, retries)
                _mark_uploading(conn, event_id)

                remote_id = row["remote_event_id"]
                if not remote_id:
                    remote_id = _create_event_on_backend(cfg, row, session)
                    _set_remote_event_id(conn, event_id, remote_id)

                if row["image_rel_path"]:
                    _upload_image(cfg, str(remote_id), str(row["image_rel_path"]), session)

                _mark_sent(conn, event_id)
                LOG.info(f"Sent | local_id={event_id} remote_id={remote_id}")

            except Exception as e:
                _mark_failed(conn, event_id, str(e))
                LOG.warning(f"Failed | id={event_id} retries={retries+1} err={e}")


# ============================================================
# Main entrypoint (service process)
# ============================================================
def _usage() -> None:
    print("Usage:")
    print("  python uploader.py <config.yaml>")
    print("")
    print("Main process usage:")
    print("  from uploader import writer_init, write_event")
    print("  writer_init('config.yaml')")
    print("  write_event(...)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _usage()
        sys.exit(1)

    run_uploader(sys.argv[1])
