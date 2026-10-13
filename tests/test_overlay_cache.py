import cv2
import numpy as np
import pytest

from ddc.dancer_features import write_synthetic_demo
from ddc.overlay_cache import overlay_key, render_dancer_overlay


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_synthetic_demo(run / "tracking.jsonl", tmp_path / "labels.csv", frames=24)
    writer = cv2.VideoWriter(str(run / "source.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (900, 800))
    for _ in range(24):
        writer.write(np.full((800, 900, 3), 40, np.uint8))
    writer.release()
    return run


def _render(run, dancers, **kw):
    return render_dancer_overlay(run, dancers, range(8), **kw)


def test_render_is_downscaled_and_playable(run):
    out = _render(run, [0, 1, 2, 3], max_height=400)
    assert not out["cached"]
    cap = cv2.VideoCapture(out["overlay"])
    ok, frame = cap.read()
    assert ok and frame.shape[0] <= 400 and frame.shape[0] % 2 == 0
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 20


def test_identical_request_is_served_from_cache(run):
    first = _render(run, [0, 1, 2, 3])
    mtime = (run / "overlays").glob("*.mp4").__next__().stat().st_mtime_ns
    second = _render(run, [3, 2, 1, 0])  # same set, different order
    assert second["cached"] and second["overlay"] == first["overlay"]
    assert (run / "overlays").glob("*.mp4").__next__().stat().st_mtime_ns >= mtime


def test_changed_dancers_merges_or_source_make_a_new_render(run):
    base = _render(run, [0, 1, 2, 3])["overlay"]
    assert _render(run, [0, 1, 2])["overlay"] != base
    assert _render(run, [0, 1, 2, 3], merge_groups=[[0, 1]])["overlay"] != base
    source = run / "source.mp4"
    k1 = overlay_key(run, source, None, [0], 720)
    source.write_bytes(source.read_bytes() + b"\0")
    assert overlay_key(run, source, None, [0], 720) != k1


def test_old_renders_are_pruned_and_no_partials_remain(run):
    for dancers in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 3]):
        _render(run, dancers, keep=2)
    files = sorted(p.name for p in (run / "overlays").iterdir())
    assert len([f for f in files if f.endswith(".mp4")]) == 2
    assert not [f for f in files if "partial" in f]


def test_jsonl_keeps_original_coordinates_and_only_dancers(run):
    import json
    out = _render(run, [0, 1], max_height=400)
    people = [p for line in open(out["json"]) for p in json.loads(line)["people"]]
    assert {p["stable_track_id"] for p in people} == {0, 1}
    # Rendered at half size these boxes would end near x=190; the JSONL keeps source pixels.
    assert max(p["bbox"][2] for p in people) > 300
