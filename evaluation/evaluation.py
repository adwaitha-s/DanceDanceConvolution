"""Local, frame-accurate dance video review. Run with python evaluation/evaluation.py."""

from __future__ import annotations

import csv
import errno
import json
import logging
import re
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime
from fractions import Fraction
from http.client import HTTPConnection, HTTPException as HTTPClientError
from pathlib import Path
from uuid import UUID, uuid4

import imageio_ffmpeg
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RUN_ID = re.compile(r"run-\d{8}-\d{6}-[a-f0-9]{8}")
NOTE_FIELDS = ("timestamp", "comment", "frame", "id")
app = FastAPI(title="DanceDanceConvolution Evaluation", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")


def run_folder(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id) or not (ROOT / run_id / "run.json").is_file():
        raise HTTPException(404, "This run could not be found.")
    return ROOT / run_id


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def timestamp(microseconds: int) -> str:
    seconds, ms = divmod((microseconds + 500) // 1000, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02}.{ms:03}"


def prepare_video(source: Path, folder: Path) -> tuple[list[int], dict]:
    """Make one seekable preview and index its actual presentation timestamps."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    preview = folder / "preview.mp4"
    try:
        subprocess.run([
            *command, "-xerror", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "mov,matroska,webm,avi,asf,mpeg,mpegts,flv,ogg",
            "-i", str(source), "-map", "0:V:0", "-map", "0:a:0?",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-fps_mode:v", "passthrough", "-enc_time_base:v", "1:1000000",
            "-c:a", "aac", "-movflags", "+faststart", str(preview),
        ], check=True, capture_output=True)
        index = subprocess.run([
            *command, "-copyts", "-i", str(preview), "-map", "0:v:0",
            "-c:v", "copy", "-f", "framecrc", "-",
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logging.getLogger(__name__).warning("Video preparation failed: %s", exc.stderr)
        raise HTTPException(422, "Could not read this video. Try a playable MP4, MOV, WebM or AVI file.") from exc

    # H.264 packets contain one frame each and may arrive out of display order.
    header = dict(line[1:].split(": ", 1) for line in index.stdout.splitlines() if line.startswith("#"))
    time_base = Fraction(header["tb 0"])
    packets = [line.split(",") for line in index.stdout.splitlines() if line.startswith("0,")]
    frames = sorted((int(p[2]), int(p[3])) for p in packets)
    if not frames:
        raise HTTPException(422, "The video contains no readable frames.")
    ticks_to_us = time_base * 1_000_000
    starts = [round(pts * ticks_to_us) for pts, _ in frames]
    last_duration = frames[-1][1]
    if last_duration <= 0:
        last_duration = frames[-1][0] - frames[-2][0] if len(frames) > 1 else 1
    end = round((frames[-1][0] + last_duration) * ticks_to_us)
    width, height = map(int, header["dimensions 0"].split("x"))
    return starts, {
        "width": width, "height": height, "end_us": end,
        "fps": round(len(starts) * 1_000_000 / max(1, end - starts[0]), 3),
    }


def read_notes(folder: Path) -> list[dict]:
    with (folder / "notes.csv").open(encoding="utf-8-sig", newline="") as f:
        return [dict(row, frame=int(row["frame"])) for row in csv.DictReader(f)]


def write_notes(folder: Path, notes: list[dict]) -> None:
    # Readers see either complete version, even if a write is interrupted.
    temp = folder / "notes.tmp"
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NOTE_FIELDS)
        writer.writeheader()
        writer.writerows(notes)
    temp.replace(folder / "notes.csv")


@app.get("/")
def home():
    return FileResponse(ASSETS / "index.html")


@app.get("/api/runs")
def list_runs():
    runs = []
    for file in ROOT.glob("run-*/run.json"):
        if RUN_ID.fullmatch(file.parent.name):
            runs.append(read_json(file))
    return sorted(runs, key=lambda run: run["created"], reverse=True)


@app.post("/api/runs", status_code=201)
def upload_run(video: UploadFile):
    name = Path((video.filename or "video").replace("\\", "/")).name
    extension = Path(name).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
        extension = ".video"
    now = datetime.now().astimezone()
    run_id = f"run-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    # Publish only after the original, preview, index and CSV all exist.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=ROOT) as staging:
        folder = Path(staging)
        source = folder / f"source{extension}"
        with source.open("wb") as dest:
            shutil.copyfileobj(video.file, dest, length=1024 * 1024)
        starts, info = prepare_video(source, folder)
        run = dict(id=run_id, filename=name, created=now.isoformat(), size=source.stat().st_size, **info)
        write_json(folder / "run.json", run)
        write_json(folder / "frames.json", starts)
        write_notes(folder, [])
        folder.rename(ROOT / run_id)
    return run


@app.get("/api/runs/{run_id}")
def open_run(run_id: str):
    folder = run_folder(run_id)
    return dict(**read_json(folder / "run.json"), frames=read_json(folder / "frames.json"),
                notes=read_notes(folder))


@app.get("/api/runs/{run_id}/video")
def preview(run_id: str):
    return FileResponse(run_folder(run_id) / "preview.mp4", media_type="video/mp4")


@app.get("/api/runs/{run_id}/notes.csv")
def download_notes(run_id: str):
    folder = run_folder(run_id)
    return FileResponse(folder / "notes.csv", media_type="text/csv", filename=f"{run_id}-notes.csv")


class Note(BaseModel):
    frame: int = Field(ge=1)
    comment: str = Field(min_length=1)


@app.put("/api/runs/{run_id}/notes/{note_id}")
def save_note(run_id: str, note_id: UUID, note: Note):
    folder = run_folder(run_id)
    starts = read_json(folder / "frames.json")
    comment = note.comment.strip()
    if not comment or note.frame > len(starts):
        raise HTTPException(422, "Select a frame and enter a comment.")
    with FileLock(str(folder / ".notes.lock")):
        notes = read_notes(folder)
        existing = next((row for row in notes if row["id"] == str(note_id)), None)
        if existing is not None:
            if existing["frame"] != note.frame:
                raise HTTPException(409, "This note belongs to a different frame.")
            existing["comment"] = comment
        else:
            notes.append(dict(id=str(note_id), frame=note.frame,
                              timestamp=timestamp(starts[note.frame - 1]), comment=comment))
        notes.sort(key=lambda row: row["frame"])
        try:
            write_notes(folder, notes)
        except PermissionError as exc:
            raise HTTPException(409, "Close notes.csv in your spreadsheet editor and try saving again.") from exc
    return notes


def evaluation_is_running(host: str, port: int) -> bool:
    connection = HTTPConnection(host, port, timeout=1)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        return response.status == 200 and f"<title>{app.title}</title>".encode() in response.read(4096)
    except (OSError, HTTPClientError):
        return False
    finally:
        connection.close()


def serve() -> None:
    host = "127.0.0.1"
    for port in range(7861, 7961):
        url = f"http://{host}:{port}"
        with socket.socket() as listener:
            try:
                listener.bind((host, port))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                if evaluation_is_running(host, port):
                    print(f"Evaluation is already running. Open {url}")
                    return
                continue
            print(f"Open the evaluation app at {url}", flush=True)
            server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
            server.run(sockets=[listener])
            return
    raise SystemExit("No available local port was found between 7861 and 7960.")


if __name__ == "__main__":
    serve()
