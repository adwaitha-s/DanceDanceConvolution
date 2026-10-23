"""Reference-pose deviation modes: dancer-as-reference and solo-video resampling."""

import numpy as np

from ddc.deviation import compute_deviation
from ddc.reference import _resample, reference_from_track


def _norm(offsets, frames=12):
    """(F,T,17,3) with dancer t's whole skeleton shifted by offsets[t] from the origin."""
    norm = np.zeros((frames, len(offsets), 17, 3))
    norm[..., 2] = 1.0
    for t, (x, y) in enumerate(offsets):
        norm[:, t, :, 0], norm[:, t, :, 1] = x, y
    return norm


def test_composite_is_default_mode():
    dev = compute_deviation(_norm([(0, 0), (0.1, 0), (0, 0.3)]))
    assert dev.mode == "composite" and dev.reference_track is None
    assert np.isfinite(dev.score).all()


def test_dancer_reference_scores_others_against_it():
    norm = _norm([(0, 0), (0.1, 0), (0, 0.3)])
    ref = reference_from_track(norm, 0)
    dev = compute_deviation(norm, reference=ref, reference_track=0)
    assert dev.mode == "dancer" and not dev.leave_one_out
    assert np.isnan(dev.score[:, 0]).all() and np.isnan(dev.joint[:, 0]).all()
    np.testing.assert_allclose(dev.score[:, 1], 0.1)
    np.testing.assert_allclose(dev.score[:, 2], 0.3)
    assert (dev.n_present == 2).all()   # reference dancer isn't counted as scored


def test_solo_reference_scores_every_dancer():
    norm = _norm([(0.2, 0), (0, 0.5)])
    ref = np.zeros((12, 17, 2))
    dev = compute_deviation(norm, reference=ref)
    assert dev.mode == "solo" and dev.reference_track is None
    np.testing.assert_allclose(dev.score[:, 0], 0.2)
    np.testing.assert_allclose(dev.score[:, 1], 0.5)


def test_resample_interpolates_and_blanks_outside_range():
    t_src = np.array([0.0, 1.0, 2.0])
    ref = np.zeros((3, 17, 2))
    ref[:, :, 0] = t_src[:, None]            # x == t
    out = _resample(ref, t_src, np.array([-0.5, 0.0, 0.25, 1.5, 2.0, 2.5]))
    assert np.isnan(out[0]).all() and np.isnan(out[5]).all()
    np.testing.assert_allclose(out[1:5, 0, 0], [0.0, 0.25, 1.5, 2.0])


def test_resample_propagates_nan_neighbours():
    t_src = np.array([0.0, 1.0, 2.0])
    ref = np.zeros((3, 17, 2))
    ref[1] = np.nan
    out = _resample(ref, t_src, np.array([0.5, 1.5]))
    assert np.isnan(out).all()


def test_time_tolerance_forgives_small_timing_offset():
    rng = np.random.default_rng(0)
    F, lag = 30, 2
    ref = rng.normal(size=(F, 17, 2))
    norm = np.ones((F, 1, 17, 3))
    norm[lag:, 0, :, :2] = ref[:-lag]     # dancer does the reference's moves 2 frames late
    norm[:lag, 0, :, :2] = ref[0]
    strict = compute_deviation(norm, reference=ref)
    loose = compute_deviation(norm, reference=ref, tolerance=lag)
    assert np.nanmean(strict.score[lag:]) > 0.5
    np.testing.assert_allclose(loose.score[lag:, 0], 0, atol=1e-9)
    assert (loose.lag[lag:, 0] == -lag).all()
    np.testing.assert_allclose(loose.ref_pose(10, 0), ref[10 - lag])


def test_zero_tolerance_matches_simultaneous_frames():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(10, 17, 2))
    norm = np.ones((10, 2, 17, 3))
    norm[..., :2] = rng.normal(size=(10, 2, 17, 2))
    a = compute_deviation(norm, reference=ref)
    b = compute_deviation(norm, reference=ref, tolerance=0)
    np.testing.assert_array_equal(a.score, b.score)
    assert b.lag is None and b.ref_pose(3, 1) is not None
