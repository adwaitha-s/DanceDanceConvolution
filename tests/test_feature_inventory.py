from ddc.dancer_features import write_synthetic_demo
from ddc.feature_inventory import FEATURE_SPECS, write_inventory_artifacts


def test_feature_inventory_writes_wide_matrix_and_embedded_predictions(tmp_path):
    tracking = tmp_path / "tracking.jsonl"
    labels = tmp_path / "labels.csv"
    out = tmp_path / "out"
    write_synthetic_demo(tracking, labels, frames=50)

    result = write_inventory_artifacts(tracking, labels, out)
    assert len(FEATURE_SPECS) >= 50
    assert len(result["selected_features"]) == 3
    assert (out / "feature_inventory.json").exists()
    assert (out / "track_feature_inventory.csv").exists()
    assert (out / "dancer_predictions.csv").exists()
    assert "Embedded" in result["selection_method"]
