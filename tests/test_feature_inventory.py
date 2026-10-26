from ddc.feature_inventory import collect_feature_inventory
from ddc.training_features import FEATURES, PREDICTOR_IDS
from test_data import write_synthetic_demo


def test_training_inventory_measures_candidates_on_merged_tracks(tmp_path):
    tracking = tmp_path / "tracking.jsonl"
    write_synthetic_demo(tracking, tmp_path / "labels.csv", frames=50)

    rows = collect_feature_inventory(tracking, [[0, 1]])
    assert len(rows) == 7
    assert [row["track_id"] for row in rows] == list(range(7))
    assert len(FEATURES) == 40
    assert all(fid in rows[0]["values"] for fid in PREDICTOR_IDS)
    assert not any(fid.startswith("context_") or fid.startswith("human_")
                   for fid in PREDICTOR_IDS)
