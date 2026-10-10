"""Write the cross-run dancer-feature registry as an Excel workbook.

Pure-Python (openpyxl) port of the former ``scripts/build_feature_registry.mjs``.
Input is the model dict from ``feature_registry.build_registry``; it consumes saved
pose statistics only and never invokes the detector.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

NAVY, TEAL, PALE, LINE = "17365D", "0F766E", "E8F1FA", "D6E2F0"
_thin = Side(style="thin", color=LINE)
_BORDER = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)


def _fill(color: str) -> PatternFill:
    return PatternFill("solid", start_color=color, end_color=color)


def _sheet_name(wb: Workbook, desired: str) -> str:
    base = "".join("-" if c in '\\/*?:[]' else c for c in desired)[:31] or "Sheet"
    name, n = base, 2
    while name in wb.sheetnames:
        name = f"{base[:28]}-{n}"
        n += 1
    return name


def _title(ws, text: str, subtext: str) -> None:
    ws.merge_cells("A1:H1")
    ws["A1"] = text
    ws["A1"].font = Font(bold=True, size=16, color="FFFFFF")
    ws["A1"].fill = _fill(NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 29
    ws.merge_cells("A2:H2")
    ws["A2"] = subtext
    ws["A2"].font = Font(italic=True, color="35506D")
    ws["A2"].fill = _fill(PALE)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[2].height = 30
    ws.sheet_view.showGridLines = False


def _header(ws, row: int, col: int, labels: list[str]) -> None:
    for i, label in enumerate(labels):
        c = ws.cell(row=row, column=col + i, value=label)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = _fill(TEAL)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.border = _BORDER


def _rows(ws, row: int, col: int, rows: list[list], formats: dict[int, str] | None = None) -> int:
    """Write bordered rows starting at (row, col); ``formats`` maps column offset -> format."""
    for r, values in enumerate(rows):
        for i, value in enumerate(values):
            c = ws.cell(row=row + r, column=col + i, value=value)
            c.border = _BORDER
            c.alignment = Alignment(vertical="center")
            if formats and i in formats:
                c.number_format = formats[i]
    return row + len(rows)


def _label_pairs(ws, row: int, pairs: list[tuple]) -> None:
    for i, (label, value) in enumerate(pairs):
        a = ws.cell(row=row + i, column=1, value=label)
        a.font = Font(bold=True, color=NAVY)
        a.fill = _fill("DCEAF7")
        a.border = _BORDER
        ws.cell(row=row + i, column=2, value=value).border = _BORDER


def _note(ws, cell_range: str, text: str, fill: str, color: str) -> None:
    ws.merge_cells(cell_range)
    c = ws[cell_range.split(":")[0]]
    c.value = text
    c.fill = _fill(fill)
    c.font = Font(color=color)
    c.alignment = Alignment(wrap_text=True, vertical="center")


def _widths(ws, widths: dict[str, float]) -> None:
    for letter, width in widths.items():
        ws.column_dimensions[letter].width = width


def write_registry_workbook(model: dict, path: str | Path) -> Path:
    path = Path(path)
    selected = model["selected_features"]
    wb = Workbook()

    # --- Registry Summary -------------------------------------------------
    ws = wb.active
    ws.title = "Registry Summary"
    _title(ws, "Accumulating Dancer Predictor Registry",
           "Cross-run standardized L1 logistic model. The checklist validates predictions when "
           "unchanged; edits become corrected training labels.")
    _label_pairs(ws, 4, [("Validated tracks", model["labelled_tracks"]),
                         ("Validated dancers", model["dancers"]),
                         ("Validated non-dancers", model["non_dancers"]),
                         ("Runs contributing labels", len(model["runs"])),
                         ("Rebuilt (UTC)", model["generated_at"])])
    _header(ws, 10, 1, ["Rank", "Top 5 predictor", "Category", "Std. coefficient",
                        "Absolute weight", "Direction"])
    end = _rows(ws, 11, 1, [[x["rank"], x["feature_name"], x["category"],
                             x["standardized_coefficient"], x["absolute_weight"], x["direction"]]
                            for x in selected], {3: "0.000", 4: "0.000"})
    _note(ws, "A18:H19",
          "Weight is a standardized logistic-regression coefficient, not a direct percentage. A "
          "positive coefficient increases the log-odds of dancer status for a one-standard-deviation "
          "feature increase; a negative coefficient decreases it. Camera-framing features (frame "
          "position, clip length, detected size, detector confidence) are excluded from the "
          "predictors because they describe the recording, not the dancing.",
          "FFF7E6", "755100")
    _note(ws, "A21:H22", model["run_top5_method"], "EAF4EF", "24543A")
    if selected:
        chart = BarChart()
        chart.type = "bar"
        chart.title = "Top 5 absolute predictor weights"
        chart.legend = None
        chart.add_data(Reference(ws, min_col=5, min_row=10, max_row=end - 1), titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=2, min_row=11, max_row=end - 1))
        ws.add_chart(chart, "H4")
    ws.freeze_panes = "A3"
    _widths(ws, {"A": 24, "B": 31, "C": 29, "D": 17, "E": 17, "F": 28})

    # --- Feature Inventory Summary ---------------------------------------
    inv = wb.create_sheet("Feature Inventory Summary")
    _title(inv, "Pose Feature Inventory and Embedded Selection",
           "Current run-local feature summary. Features reuse saved tracking JSON; pose detection "
           "was not rerun.")
    _label_pairs(inv, 4, [("Selection method", "Embedded L1 logistic regression"),
                          ("Labeled tracks", model["labelled_tracks"]),
                          ("Dancers", model["dancers"]),
                          ("Non-dancers", model["non_dancers"]),
                          ("Selected predictors", len(selected)),
                          ("Training basis", "All validated tracks in the registry")])
    _header(inv, 11, 1, ["Rank", "Selected feature", "Category", "Absolute influence", "Coefficient"])
    _rows(inv, 12, 1, [[x["rank"], x["feature_name"], x["category"], x["absolute_weight"],
                        x["standardized_coefficient"]] for x in selected], {3: "0.000", 4: "0.000"})
    _note(inv, "A19:H20",
          "Human-related fields are documented in Feature Catalog. Age, gender identity, "
          "health/injury, and fatigue are not inferred from video and are excluded from "
          "classification unless voluntarily supplied through an appropriate consented data source.",
          "FFF7E6", "755100")
    _widths(inv, {"A": 22, "B": 32, "C": 30, "D": 18, "E": 18})

    # --- Feature Catalog --------------------------------------------------
    cat = wb.create_sheet("Feature Catalog")
    _title(cat, "Feature Catalog", "Definitions, source signals, units, and eligibility. Selected "
           "ranks refer to the current cumulative registry model.")
    _header(cat, 4, 1, ["Feature ID", "Category", "Feature", "Definition", "Unit", "Source signal",
                        "Predictor eligible", "Availability", "Registry rank"])
    rank = {x["feature_id"]: x["rank"] for x in selected}
    _rows(cat, 5, 1, [[x["id"], x["category"], x["name"], x["definition"], x["unit"],
                       x["source_signal"], "Yes" if x["predictor_eligible"] else "No",
                       x["availability"], rank.get(x["id"])] for x in model["feature_catalog"]])
    cat.freeze_panes = "A5"
    _widths(cat, {"A": 31, "B": 26, "C": 28, "D": 52, "E": 24, "F": 33, "G": 18, "H": 36, "I": 14})

    # --- One Labels sheet per run, one detail sheet per validated dancer --
    for run in model["runs_detail"]:
        validation = run["validation"]
        status = validation.get("status") or ("historical_labels" if run["labelled_track_count"]
                                              else "awaiting_checklist")
        corrected = set(validation.get("corrected_track_ids", []))
        sheet = wb.create_sheet(_sheet_name(wb, f"Labels-{run['run']}"))
        _title(sheet, f"Labels — {run['run']}",
               "One row per stabilized person track. This is the reusable review sheet for the run; "
               "labels and probabilities feed the cumulative predictor after validation.")
        _header(sheet, 4, 1, ["Track ID", "Frames present", "Reviewed label", "Predicted label",
                              "Dancer probability", "Validation status", "Correction recorded"])
        _rows(sheet, 5, 1, [[t["track_id"], t["frames_present"], t.get("reviewed_label", ""),
                             t["predicted_label"], t["dancer_probability"],
                             status.replace("_", " "), "Yes" if t["track_id"] in corrected else ""]
                            for t in run["tracks"]], {4: "0.0%"})
        _header(sheet, 4, 9, ["Registry top 5 used for prediction", "Std. coefficient",
                              "Absolute weight", "Direction"])
        top5 = run.get("scoring_top5") or run.get("run_top5") or []
        local = ([[x["feature_name"], x["standardized_coefficient"], x["absolute_weight"],
                   x["direction"]] for x in top5]
                 or [["Awaiting at least two dancer and two non-dancer labels", None, None, None]])
        _rows(sheet, 5, 9, local, {1: "0.000", 2: "0.000"})
        sheet.freeze_panes = "A5"
        _widths(sheet, {"A": 13, "B": 16, "C": 18, "D": 18, "E": 20, "F": 18, "G": 18,
                        "I": 33, "J": 18, "K": 18, "L": 22})

        for track in (t for t in run["tracks"] if t.get("reviewed_label") == "dancer"):
            d = wb.create_sheet(_sheet_name(wb, f"D{track['track_id'] + 1}-{run['run']}"))
            _title(d, f"Run {run['run']} — dancer track {track['track_id'] + 1}",
                   f"Validated dancer. Model probability: {track['dancer_probability'] * 100:.1f}%. "
                   "Top five global predictors are shown with this track's contribution.")
            _label_pairs(d, 4, [("Frames present", track["frames_present"]),
                                ("Predicted probability", track["dancer_probability"]),
                                ("Predicted label", track["predicted_label"]),
                                ("Validated label", track["reviewed_label"])])
            d["B5"].number_format = "0.0%"
            _header(d, 9, 1, ["Rank", "Predictor", "Category", "Raw value", "Std. value",
                              "Std. coefficient", "Track logit contribution"])
            _rows(d, 10, 1, [[p["rank"], p["feature_name"], p["category"], p["raw_value"],
                              p["standardized_value"], p["standardized_coefficient"],
                              p["logit_contribution"]] for p in track["top_predictors"]],
                  {i: "0.000" for i in range(3, 7)})
            d.freeze_panes = "A10"
            _widths(d, {get_column_letter(i): 20 for i in range(1, 8)} | {"B": 32, "C": 30})

    wb.save(path)
    return path
