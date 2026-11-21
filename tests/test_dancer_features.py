import pytest

from ddc.dancer_features import load_labels


def test_load_human_reviewed_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("track_id,label\n0,dancer\n1,non_dancer\n2,yes\n3,bystander\n")
    assert load_labels(path) == {0: 1, 1: 0, 2: 1, 3: 0}


def test_reject_unknown_label(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("track_id,label\n0,maybe\n")
    with pytest.raises(ValueError, match="unknown label"):
        load_labels(path)
