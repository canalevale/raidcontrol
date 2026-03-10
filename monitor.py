# monitor.py
import os
import sqlite3
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

DB_PATH = "./local_db/uploader.db"
IMAGES_DIR = "./local_db/images"

app = FastAPI()

api = APIRouter(prefix="/api")


def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@api.get("/events")
def api_events(limit: int = 50):
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT id, local_uuid, station_id, detected_at, number_str,
                   image_rel_path, meta_json, status, retries, last_error
            FROM events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        counts_rows = conn.execute(
            "SELECT status, COUNT(*) as c FROM events GROUP BY status"
        ).fetchall()
        counts = {r["status"]: int(r["c"]) for r in counts_rows}
        for k in ["pending", "uploading", "sent", "failed"]:
            counts.setdefault(k, 0)

        return JSONResponse({"events": [dict(r) for r in rows], "counts": counts})
    finally:
        conn.close()


@api.get("/image")
def api_image(rel: str):
    images_dir = os.path.abspath(IMAGES_DIR)
    abs_path = os.path.abspath(os.path.join(images_dir, rel))

    # Security: keep inside images_dir
    if not abs_path.startswith(images_dir):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.exists(abs_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(abs_path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")


app.include_router(api)

# IMPORTANT: mount static *after* routes
app.mount("/", StaticFiles(directory="web", html=True), name="web")

