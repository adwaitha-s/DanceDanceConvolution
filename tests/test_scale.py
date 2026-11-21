"""Scale foundations: pose cache, memoized tracking, and vectorized math matching the loops."""

import json
import os
import warnings

import numpy as np
import pytest

from ddc import io as ddc_io
from ddc import tracking
from ddc.composite import _batch_median, _smooth_gaps, build_composite, geometric_median
from ddc.deviation import _match_in_window
from ddc.palette import DANCER_COLORS, dancer_bgr
from ddc.plots import _bin
from ddc.skeleton import BODY
from ddc.smoothing import nan_moving_mean
from test_ddc import base_pose, place


def _write_run(path, n_frames=60, xs=(200, 500)):
    with open(path, "w") as f:
        for fr in range(n_frames):
            people = []
            for i, x in enumerate(xs):
                p = place(base_pose(), x + fr, 300, 40)
                people.append({"id": i, "bbox": [x - 50.0, 200.0, x + 50.0, 420.0], "conf": 0.9,
                               "keypoints": {n: p[j].tolist() for j, n in enumerate(BODY)},
                               "left_hand": {"x": [1, 2]}})
            f.write(json.dumps({"frame": fr, "t": fr / 30, "people": people}) + "\n")


def _loop_moving_mean(x, win):
    out = np.full_like(x, np.nan)
    h = win // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg, axis=0)
    return out


def _same(a, b, atol=1e-9):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=atol)


def test_pose_cache_round_trips_and_invalidates(tmp_path):
    jl = tmp_path / "tracking.jsonl"
    _write_run(jl)
    first = ddc_io.load_jsonl(jl)
    assert ddc_io.cache_path(jl).exists()
    second = ddc_io.load_jsonl(jl)
    assert second._raw is None                       # came from the cache, raw not parsed
    assert np.array_equal(first.t, second.t)
    for a, b in zip(first.kps, second.kps):
        assert _same(a, b)
    assert np.allclose(second.bbox[0][1], [450, 200, 550, 420]) and second.conf[0][0] == 0.9
    assert second.raw[0][0]["left_hand"] == {"x": [1, 2]}   # lazily parsed on demand

    _write_run(jl, n_frames=30)
    st = jl.stat()
    os.utime(jl, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert ddc_io.load_jsonl(jl).n_frames == 30      # stale cache ignored


def test_stable_track_ids_are_cached_on_disk(tmp_path):
    jl = tmp_path / "tracking.jsonl"
    _write_run(jl)
    det = ddc_io.load_jsonl(jl)
    want_ids, want_n = tracking.stabilize(det.kps, fps=30.0, boxes=det.bbox)
    ids, n, info = tracking.stable_track_ids(det, return_info=True)
    assert n == want_n == 2 and all((a == b).all() for a, b in zip(ids, want_ids))
    assert (tmp_path / "tracking.tracks.npz").exists()
    ids[0][:] = -5                                   # callers get copies
    tracking._TRACK_CACHE.clear()
    again, n2, info2 = tracking.stable_track_ids(ddc_io.load_jsonl(jl), return_info=True)  # disk
    assert n2 == n and all((a == b).all() for a, b in zip(again, want_ids))
    assert info2["fragments"] == info["fragments"]


@pytest.mark.parametrize("shape", [(80,), (120, 3, 2), (5, 2)])
@pytest.mark.parametrize("win", [1, 2, 3, 9, 15])
def test_nan_moving_mean_matches_loop(shape, win):
    rng = np.random.default_rng(win)
    x = rng.normal(400, 80, shape)
    x[rng.random(shape) < 0.3] = np.nan
    x[2:4] = np.nan
    assert _same(nan_moving_mean(x, win), _loop_moving_mean(x, win))


def _exact_median(P, W):
    from scipy.optimize import minimize
    obj = lambda x: (W * np.linalg.norm(P - x, axis=1)).sum()
    cands = [minimize(obj, x0, method="Nelder-Mead",
                      options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000)).x
             for x0 in [P.mean(0)] + list(P)]
    return min(cands + list(P), key=obj)


def test_batch_median_finds_the_true_median():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 1, (300, 6, 2))
    w = rng.uniform(0.3, 1, (300, 6))
    w[::3, 0] = 4.0                                  # a dominant point: the median sits on it
    valid = rng.random((300, 6)) < 0.7
    valid[:5] = False
    valid[5:10] = False
    valid[5:10, :1] = True                           # 1 and 2 valid points: special cases
    valid[10:15, :2] = True
    got = _batch_median(pts, valid, w)
    for b in range(300):
        P, W = pts[b][valid[b]], w[b][valid[b]]
        if len(P) == 0:
            assert np.isnan(got[b]).all()
        elif len(P) <= 2:
            assert np.allclose(got[b], P.mean(0))
        else:
            assert np.allclose(got[b], _exact_median(P, W), atol=1e-4)
            assert np.allclose(geometric_median(P, W), got[b], atol=1e-6)


def test_warm_start_on_a_data_point_still_finds_the_median():
    pts = np.array([[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]])
    w = np.ones((1, 4))
    valid = np.ones((1, 4), bool)
    got = _batch_median(pts, valid, w, x0=np.array([[0.0, 0.0]]))   # start exactly on a point
    assert np.allclose(got[0], [5.0, 5.0], atol=1e-4)


def test_smooth_gaps_only_fills_short_gaps():
    c = np.tile(np.arange(40, dtype=float)[:, None, None], (1, 17, 2))
    c[5:8] = np.nan        # 3-frame gap: interpolated
    c[20:32] = np.nan      # 12-frame gap: left empty
    out = _smooth_gaps(c, strength=1, max_gap=8)
    assert np.allclose(out[5:8, 0, 0], [5, 6, 7]) and np.isnan(out[20:32]).all()


def test_leave_one_out_composite_matches_per_frame_definition():
    rng = np.random.default_rng(2)
    norm = np.concatenate([rng.normal(0, 0.2, (30, 5, 17, 2)), rng.uniform(0.2, 1, (30, 5, 17, 1))], -1)
    norm[3:15, 2] = np.nan                           # dancer 2 absent longer than the 8-frame fill
    loo = build_composite(norm, smooth=1, leave_one_out=True)
    for f in (0, 4, 20):
        for t in range(5):
            if np.isnan(norm[f, t, :, 0]).all():
                assert np.isnan(loo[f, t]).all()
                continue
            for j in (0, 9):
                keep = [u for u in range(5) if u != t and np.isfinite(norm[f, u, j, 0]) and norm[f, u, j, 2] >= 0.3]
                P, W = norm[f, keep, j, :2], norm[f, keep, j, 2]
                want = (np.full(2, np.nan) if not keep else P.mean(0) if len(keep) <= 2
                        else _exact_median(P, W))
                assert _same(loo[f, t, j], want, atol=1e-4)


def _loop_match(xy, ok, ref, tolerance):
    def shift(r, s):
        out = np.full_like(r, np.nan)
        F = len(r)
        if s >= 0:
            out[:F - s] = r[s:]
        else:
            out[-s:] = r[:F + s]
        return out
    F, T = xy.shape[:2]
    best = np.full((F, T), np.inf)
    matched = np.full_like(ref, np.nan)
    lag = np.zeros((F, T), int)
    for s in sorted(range(-tolerance, tolerance + 1), key=lambda s: (abs(s), s)):
        cand = shift(ref, s)
        dist = np.linalg.norm(xy - cand, axis=-1)
        dist[~ok] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = np.nanmean(dist, axis=-1)
        better = np.isfinite(score) & (score < best)
        best[better] = score[better]
        matched[better] = cand[better]
        lag[better] = s
    return matched, lag


def test_match_in_window_matches_loop_version():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, (50, 4, 17, 2))
    xy = np.roll(ref, 2, axis=0) + rng.normal(0, 0.05, ref.shape)
    ok = rng.random((50, 4, 17)) < 0.8
    ok[10:12] = False
    ref[30:33] = np.nan
    got_m, got_l = _match_in_window(xy, ok, ref, 3)
    want_m, want_l = _loop_match(xy, ok, ref, 3)
    assert np.array_equal(got_l, want_l) and _same(got_m, want_m)


def test_bin_and_palette():
    x = np.array([1.0, 3.0, np.nan, np.nan, 5.0])
    assert _same(_bin(x, 2), np.array([2.0, np.nan, 5.0]))
    assert _bin(x, 1) is x
    assert len(set(DANCER_COLORS)) == 30
    assert dancer_bgr(0) == (0x3c, 0xa0, 0xf5)
