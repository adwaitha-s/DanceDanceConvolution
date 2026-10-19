"""Coaching eligibility, formula, signed feedback and analysis integration."""
import csv

import numpy as np
import pytest

from ddc.coaching import build_coaching, feedback_moments, joint_feedback, similarity
from ddc.deviation import compute_deviation
from ddc.skeleton import J


def poses(frames=6, dancers=2):
    p = np.zeros((frames, dancers, 17, 3))
    p[..., 2] = .9
    return p


def solo(p):
    dev = compute_deviation(p, reference=np.zeros((len(p), 17, 2)), smooth=1)
    return dev, build_coaching(dev, p)


def test_formula_bounds_monotonicity_and_invalid_values():
    scores = similarity([0, .25, 1, 100, np.nan, np.inf, -1])
    np.testing.assert_allclose(scores[:3], [100, 80, 50])
    assert np.all(np.diff(scores[:4]) < 0)
    assert np.all((scores[:4] >= 0) & (scores[:4] <= 100))
    assert np.isnan(scores[4:]).all()


def test_dancer_mean_then_mapping_and_equal_dancer_overall():
    p = poses(frames=2)
    p[1, 0, :, 0] = 1
    p[1, 1] = np.nan
    _, c = solo(p)
    assert c.dancers[0]['score'] == pytest.approx(100 / 1.5)
    assert c.dancers[1]['score'] == 100
    assert c.overall == pytest.approx((100 / 1.5 + 100) / 2)
    assert c.dancers[1]['coverage'] == .5


@pytest.mark.parametrize('loo', [True, False])
def test_composite_requires_peer_even_with_smoothing(loo):
    p = poses()
    p[2:4, 1] = np.nan
    dev = compute_deviation(p, smooth=5, leave_one_out=loo)
    c = build_coaching(dev, p)
    assert np.isnan(c.frame_score[2:4]).all()
    assert c.dancers[0]['valid_frames'] == 4
    lone = poses(dancers=1)
    c = build_coaching(compute_deviation(lone, leave_one_out=loo), lone)
    assert c.overall is None
    assert c.dancers[0]['score'] is None


def test_composite_masks_self_only_joint():
    p = poses()
    p[:, 1, J['left_wrist']] = np.nan
    dev = compute_deviation(p)
    c = build_coaching(dev, p)
    assert np.isnan(c.joint[:, :, J['left_wrist']]).all()
    assert c.overall == 100


def test_reference_excluded_and_dropout_not_scored():
    p = poses(dancers=3)
    p[:, 1, :, 0] = .25
    p[:, 2, :, 0] = 1
    p[2, 0] = np.nan
    # A smoothed reference may still exist at frame 2.
    dev = compute_deviation(p, reference=np.zeros((6, 17, 2)), reference_track=0)
    c = build_coaching(dev, p)
    assert c.dancers[0]['reference']
    assert c.dancers[0]['score'] is None
    assert np.isnan(c.frame_score[2]).all()
    assert c.dancers[1]['score'] == 80
    assert c.dancers[2]['score'] == 50
    assert c.overall == 65


def test_missing_reference_low_confidence_nonfinite_and_minimum_joints():
    p = poses(frames=4)
    p[0, :, 5:, 2] = .1  # only 5 confident joints
    p[1, :, 6:, 2] = .1  # 6 is enough
    p[2, :, :, 2] = np.nan
    p[3] = np.inf
    _, c = solo(p)
    assert np.isnan(c.frame_score[[0, 2, 3]]).all()
    assert (c.frame_score[1] == 100).all()
    assert c.dancers[0]['valid_frames'] == 1
    dev = compute_deviation(poses(), reference=np.full((6, 17, 2), np.nan))
    assert build_coaching(dev, poses()).overall is None


@pytest.mark.parametrize('offset,word', [(.5, 'lower'), (-.5, 'higher')])
def test_direction_timestamp_joint_and_region(offset, word):
    p = poses()
    j = J['left_wrist']
    p[:, 0, j, 1] = offset
    dev, c = solo(p)
    message = joint_feedback(dev, p, c, 0, 0, j, 12.5)
    assert f'left wrist is {word}' in message
    assert '12.50 seconds' in message
    assert 'left wrist' in c.dancers[0]['improvement']
    rows = feedback_moments(dev, p, c, np.arange(6.) + 12.5)
    assert len(rows) == 3  # the identical dancer has no substantial differences
    assert all(row[0] == 0 for row in rows)


def test_no_invented_direction_when_horizontal_rotated_or_missing():
    p = poses()
    j = J['right_wrist']
    p[:, 0, j, 0] = .5
    dev, c = solo(p)
    assert 'differs' in joint_feedback(dev, p, c, 0, 0, j, 0)
    p[:, 0, j, 1] = 1
    dev, c = solo(p)
    message = joint_feedback(dev, p, c, 0, 0, j, 0, rotate=True)
    assert 'differs' in message and 'lower' not in message
    assert 'close' in joint_feedback(dev, p, c, 0, 1, j, 0)
    c.joint[0, 0, j] = np.nan
    assert 'Insufficient' in joint_feedback(dev, p, c, 0, 0, j, 0)


def test_two_person_composite_symmetry_and_leave_one_out_direction():
    p = poses(dancers=3)
    p[:, 0, J['left_wrist'], 1] = .6
    dev = compute_deviation(p, leave_one_out=True, smooth=1)
    c = build_coaching(dev, p)
    assert 'lower' in joint_feedback(dev, p, c, 0, 0, J['left_wrist'], 0)
    pair = p[:, :2]
    c = build_coaching(compute_deviation(pair), pair)
    assert c.dancers[0]['score'] == c.dancers[1]['score']


def test_peak_times_separation_and_perfect_match_no_feedback():
    p = poses(frames=30)
    p[:, 0, J['left_wrist'], 1] = .5
    dev, c = solo(p)
    t = np.arange(30) * .1
    rows = feedback_moments(dev, p, c, t)
    assert len(rows) == 3
    assert np.all(np.diff([row[1] for row in rows]) >= 1)
    assert {row[1] for row in rows} <= set(t)
    dev, c = solo(poses())
    assert feedback_moments(dev, poses(), c, np.arange(6.)) == []


def test_analyze_csv_retains_raw_columns_and_adds_similarity(tmp_path):
    from ddc.analyze import analyze_run
    from test_ddc import _write_jsonl_run, base_pose, place
    run = _write_jsonl_run(tmp_path, [[place(base_pose(), 200, 300, 40),
                                     place(base_pose(), 500, 300, 40)] for _ in range(40)])
    result = analyze_run(run, render=False, reference_mode='dancer', reference_dancer=0)
    assert result.coaching.overall == pytest.approx(100)
    assert result.coaching.dancers[0]['score'] is None
    with open(result.csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40
    assert all(row['dancer'] == '2' for row in rows)
    assert float(rows[0]['score']) == pytest.approx(0)
    assert float(rows[0]['movement_similarity']) == pytest.approx(100)


def test_key_moments_rank_sustained_events_and_keep_peak_and_range():
    from ddc.coaching import key_moments
    p = poses(frames=50)
    p[2:8, 1, J['left_wrist'], 1] = .4
    p[25:33, 1, J['right_ankle'], 0] = .9
    moments = key_moments(p, np.arange(50) / 10)
    assert len(moments) == 2
    assert moments[0].dancers == (0, 1)
    assert (moments[0].start, moments[0].end, moments[0].peak) == (2.5, 3.2, 2.5)
    assert moments[0].region == 'legs' and moments[0].joint == 'right_ankle'
    assert moments[0].comparable_joints == 17
    assert moments[0].mean_deviation == pytest.approx(.9 / 17)
    assert moments[1].region == 'arms' and moments[1].joint == 'left_wrist'
    assert moments[1].mean_deviation < moments[0].mean_deviation


def test_key_moments_ignore_spikes_matches_and_single_dancers():
    from ddc.coaching import key_moments
    p = poses(frames=20)
    p[2, 1, :, 0] = 10  # one-frame detector spike is not a sustained mismatch
    assert key_moments(p, np.arange(20) / 10) == []
    assert key_moments(poses(frames=20), np.arange(20) / 10) == []
    assert key_moments(p[:, :1], np.arange(20) / 10) == []
    assert key_moments(p, np.arange(20) / 10, top_n=0) == []


def test_key_moments_missing_and_unreliable_joints_break_events():
    from ddc.coaching import key_moments
    p = poses(frames=12)
    p[:, 1, :, 0] = 1
    # Only one short continuous overlap remains in each case.
    p[2:10, 1] = np.nan
    assert key_moments(p, np.arange(12) / 10) == []
    p = poses(frames=12)
    p[:, 1, :, 0] = 1
    p[:, 0, 5:, 2] = .1  # < 6 shared confident joints
    assert key_moments(p, np.arange(12) / 10) == []
    p[:, 0, 5, 2] = .9
    moments = key_moments(p, np.arange(12) / 10)
    assert len(moments) == 1 and moments[0].comparable_joints == 6
    p[:, 0, :, 2] = np.nan
    assert key_moments(p, np.arange(12) / 10) == []


def test_key_moments_do_not_bridge_large_time_gaps():
    from ddc.coaching import key_moments
    p = poses(frames=6)
    p[:, 1, :, 0] = 1
    assert key_moments(p, [.0, .1, .2, 8., 8.1, 8.2]) == []


def test_key_moments_separate_simultaneous_pairs_and_apply_top_n():
    from ddc.coaching import key_moments
    p = poses(frames=40, dancers=3)
    p[2:8, 1, :, 0] = .5
    p[2:8, 2, :, 0] = 1
    p[25:33, 1, :, 0] = .3
    moments = key_moments(p, np.arange(40) / 10)
    assert len(moments) == 2
    assert moments[0].dancers == (0, 2)
    assert abs(moments[0].peak - moments[1].peak) >= 1
    assert len(key_moments(p, np.arange(40) / 10, top_n=1)) == 1
    assert 'early' not in repr(moments) and 'late' not in repr(moments)


@pytest.mark.parametrize('t', [[0, 0, 1], [0, np.nan, 1], [1, 0, 2], [0, 1]])
def test_key_moments_reject_invalid_timestamps(t):
    from ddc.coaching import key_moments
    with pytest.raises(ValueError, match='timestamps'):
        key_moments(poses(frames=3), t)
