import csv
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import imageio_ffmpeg
from fastapi.testclient import TestClient

from evaluation import evaluation


def test_variable_rate_review_survives_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    client = TestClient(evaluation.app)
    source = tmp_path / "rehearsal.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=size=96x64:rate=10:duration=1",
        "-vf", "setpts='if(lt(N,4),N,4+(N-4)*2)'", "-fps_mode", "vfr",
        "-c:v", "libx264", str(source),
    ], check=True)
    upload = {"video": (source.name, source.read_bytes(), "video/mp4")}
    first, second = [client.post("/api/runs", files=upload) for _ in range(2)]
    assert first.status_code == second.status_code == 201
    first_url, second_url = ["/api/runs/" + res.json()["id"] for res in (first, second)]
    assert first_url != second_url
    assert client.get(first_url).json()["frames"] == [
        0, 100000, 200000, 300000, 400000, 600000, 800000, 1000000, 1200000, 1400000
    ]
    assert client.get(first_url + "/video", headers={"range": "bytes=0-31"}).status_code == 206

    note_url = first_url + "/notes/" + str(uuid4())
    note = {"frame": 8, "comment": 'Turn, then hold.\nCheck "landing".'}
    for _ in range(2):
        assert len(client.put(note_url, json=note).json()) == 1
    note["comment"] += " ✓"
    assert client.put(note_url, json=note).status_code == 200

    def add_note(frame):
        return client.put(first_url + "/notes/" + str(uuid4()),
                          json={"frame": frame, "comment": "Check timing"}).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(add_note, [6, 8])) == [200, 200]

    reopened = TestClient(evaluation.app)
    assert len(reopened.get("/api/runs").json()) == 2
    assert reopened.get(second_url).json()["notes"] == []
    notes = reopened.get(first_url).json()["notes"]
    assert [row["frame"] for row in notes] == [6, 8, 8]
    stored = tmp_path / first.json()["id"]
    assert (stored / "source.mp4").read_bytes() == source.read_bytes()
    download = reopened.get(first_url + "/notes.csv")
    assert download.content == (stored / "notes.csv").read_bytes()
    csv_rows = list(csv.DictReader(io.StringIO(download.content.decode("utf-8-sig"))))
    assert csv_rows[1] == {**notes[1], "frame": "8", "timestamp": "00:00:01.000", "comment": note["comment"]}


def test_failed_upload_leaves_no_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    client = TestClient(evaluation.app)
    response = client.post("/api/runs", files={"video": ("broken.mp4", b"unreadable")})
    assert response.status_code == 422
    assert client.get("/api/runs").json() == []
    assert list(tmp_path.iterdir()) == []
