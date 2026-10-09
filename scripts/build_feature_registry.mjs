/* Build the cross-run dancer-feature workbook from JSON prepared by Python.
   This consumes saved pose statistics only; it never invokes the detector. */
import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const runsDir = path.resolve(process.argv[2] || "runs");
const payload = JSON.parse(await fs.readFile(path.join(runsDir, "feature_registry_workbook_payload.json"), "utf8"));
const outputPath = path.join(runsDir, "dancer_feature_registry.xlsx");
const previewPath = path.join(runsDir, "dancer_feature_registry_preview.png");
const font = "Aptos";
const navy = "#17365D", teal = "#0F766E", pale = "#E8F1FA", line = "#D6E2F0";

function uniqueSheetName(workbook, desired) {
  const base = desired.replace(/[\\/*?:\[\]]/g, "-").slice(0, 31) || "Sheet";
  let name = base, n = 2;
  while (workbook.worksheets.items.some((s) => s.name === name)) {
    name = `${base.slice(0, 28)}-${n++}`;
  }
  return name;
}

function title(sheet, text, subtext) {
  sheet.mergeCells("A1:H1");
  sheet.getRange("A1").values = [[text]];
  sheet.getRange("A1:H1").format = { fill: navy, font: { name: font, bold: true, size: 16, color: "#FFFFFF" }, verticalAlignment: "center" };
  sheet.getRange("A1:H1").format.rowHeight = 29;
  sheet.mergeCells("A2:H2");
  sheet.getRange("A2").values = [[subtext]];
  sheet.getRange("A2:H2").format = { fill: pale, font: { name: font, color: "#35506D", italic: true }, wrapText: true };
  sheet.getRange("A2:H2").format.rowHeight = 30;
  sheet.showGridLines = false;
  sheet.getRange("A:H").format.font = { name: font, size: 10 };
}

function header(range) {
  range.format = { fill: teal, font: { name: font, bold: true, color: "#FFFFFF" }, verticalAlignment: "center", wrapText: true,
                   borders: { preset: "all", style: "thin", color: "#FFFFFF" } };
}

function grid(range) {
  range.format.borders = { preset: "all", style: "thin", color: line };
  range.format.verticalAlignment = "center";
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Registry Summary");
title(summary, "Accumulating Dancer Predictor Registry", "Cross-run standardized L1 logistic model. The checklist validates predictions when unchanged; edits become corrected training labels.");
summary.getRange("A4:B8").values = [
  ["Validated tracks", payload.labelled_tracks],
  ["Validated dancers", payload.dancers],
  ["Validated non-dancers", payload.non_dancers],
  ["Runs contributing labels", payload.runs.length],
  ["Rebuilt (UTC)", payload.generated_at],
];
summary.getRange("A4:A8").format = { fill: "#DCEAF7", font: { name: font, bold: true, color: navy } };
grid(summary.getRange("A4:B8"));
summary.getRange("B8").format.numberFormat = "yyyy-mm-dd hh:mm";
summary.getRange("A10:F10").values = [["Rank", "Top 5 predictor", "Category", "Std. coefficient", "Absolute weight", "Direction"]];
header(summary.getRange("A10:F10"));
summary.getRangeByIndexes(10, 0, payload.selected_features.length, 6).values = payload.selected_features.map((x) => [
  x.rank, x.feature_name, x.category, x.standardized_coefficient, x.absolute_weight, x.direction,
]);
grid(summary.getRangeByIndexes(10, 0, payload.selected_features.length, 6));
summary.getRange(`D11:E${10 + payload.selected_features.length}`).format.numberFormat = "0.000";
summary.getRange("A18:F18").values = [["Presence / position feature", "Selected in top 5?", "Std. coefficient", "Interpretation", "", ""]];
header(summary.getRange("A18:F18"));
const contextIds = ["context_frames_present", "context_median_x_fraction", "context_median_y_fraction", "context_center_distance"];
const coefficientById = Object.fromEntries(payload.feature_ids.map((id, i) => [id, payload.coefficient[i]]));
const contextRows = contextIds.map((id) => {
  const selected = payload.selected_features.find((x) => x.feature_id === id);
  const name = ({ context_frames_present: "Frames present", context_median_x_fraction: "Horizontal frame position", context_median_y_fraction: "Vertical frame position", context_center_distance: "Distance from frame centre" })[id];
  const coefficient = coefficientById[id] ?? 0;
  return [name, selected ? "Yes" : "No", coefficient, coefficient >= 0 ? "Higher standardized value raises dancer probability" : "Higher standardized value lowers dancer probability", "", ""];
});
summary.getRangeByIndexes(18, 0, contextRows.length, 6).values = contextRows;
grid(summary.getRange("A19:F22"));
summary.getRange("C19:C22").format.numberFormat = "+0.000;-0.000;0.000";
summary.mergeCells("A24:H25");
summary.getRange("A24").values = [["Weight is a standardized logistic-regression coefficient, not a direct percentage. A positive coefficient increases the log-odds of dancer status for a one-standard-deviation feature increase; a negative coefficient decreases it. Frame-position effects can reflect camera framing, so retain them for inspection and monitor them for bias."]];
summary.getRange("A24:H25").format = { fill: "#FFF7E6", font: { name: font, color: "#755100" }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: "#E7C980" } };
summary.mergeCells("A27:H28");
summary.getRange("A27").values = [[payload.run_top5_method]];
summary.getRange("A27:H28").format = { fill: "#EAF4EF", font: { name: font, color: "#24543A" }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: "#A9D4B6" } };
if (payload.selected_features.length) {
  const chart = summary.charts.add("bar", [summary.getRange(`B10:B${10 + payload.selected_features.length}`), summary.getRange(`E10:E${10 + payload.selected_features.length}`)]);
  chart.title = "Top 5 absolute predictor weights";
  chart.titleTextStyle.typeface = font;
  chart.hasLegend = false;
  chart.setPosition("H4", "O19");
}
summary.freezePanes.freezeRows(2);
summary.getRange("A:A").format.columnWidth = 20;
summary.getRange("B:B").format.columnWidth = 31;
summary.getRange("C:C").format.columnWidth = 29;
summary.getRange("D:E").format.columnWidth = 17;
summary.getRange("F:F").format.columnWidth = 28;

// Preserve the source inventory summary and catalog in the registry workbook.
// The catalog is regenerated from the current feature definitions so newly
// added presence/position fields travel with the originally documented set.
const inventorySummary = workbook.worksheets.add("Feature Inventory Summary");
title(inventorySummary, "Pose Feature Inventory and Embedded Selection", "Current run-local feature summary. Features reuse saved tracking JSON; pose detection was not rerun.");
inventorySummary.getRange("A4:B9").values = [
  ["Selection method", "Embedded L1 logistic regression"],
  ["Labeled tracks", payload.labelled_tracks],
  ["Dancers", payload.dancers],
  ["Non-dancers", payload.non_dancers],
  ["Selected predictors", payload.selected_features.length],
  ["Training basis", "All validated tracks in the registry"],
];
inventorySummary.getRange("A4:A9").format = { fill: "#DCEAF7", font: { name: font, bold: true, color: navy } };
grid(inventorySummary.getRange("A4:B9"));
inventorySummary.getRange("A11:E11").values = [["Rank", "Selected feature", "Category", "Absolute influence", "Coefficient"]];
header(inventorySummary.getRange("A11:E11"));
inventorySummary.getRangeByIndexes(11, 0, payload.selected_features.length, 5).values = payload.selected_features.map((x) => [x.rank, x.feature_name, x.category, x.absolute_weight, x.standardized_coefficient]);
grid(inventorySummary.getRangeByIndexes(11, 0, payload.selected_features.length, 5));
inventorySummary.getRange(`D12:E${11 + payload.selected_features.length}`).format.numberFormat = "0.000";
inventorySummary.mergeCells("A19:H20");
inventorySummary.getRange("A19").values = [["Human-related fields are documented in Feature Catalog. Age, gender identity, health/injury, and fatigue are not inferred from video and are excluded from classification unless voluntarily supplied through an appropriate consented data source."]];
inventorySummary.getRange("A19:H20").format = { fill: "#FFF7E6", font: { name: font, color: "#755100" }, wrapText: true, verticalAlignment: "center" };
inventorySummary.getRange("A:A").format.columnWidth = 22;
inventorySummary.getRange("B:B").format.columnWidth = 32;
inventorySummary.getRange("C:C").format.columnWidth = 30;
inventorySummary.getRange("D:E").format.columnWidth = 18;

const catalog = workbook.worksheets.add("Feature Catalog");
title(catalog, "Feature Catalog", "Definitions, source signals, units, and eligibility. Selected ranks refer to the current cumulative registry model.");
catalog.getRange("A4:I4").values = [["Feature ID", "Category", "Feature", "Definition", "Unit", "Source signal", "Predictor eligible", "Availability", "Registry rank"]];
header(catalog.getRange("A4:I4"));
const registryRank = Object.fromEntries(payload.selected_features.map((x) => [x.feature_id, x.rank]));
catalog.getRangeByIndexes(4, 0, payload.feature_catalog.length, 9).values = payload.feature_catalog.map((x) => [
  x.id, x.category, x.name, x.definition, x.unit, x.source_signal,
  x.predictor_eligible ? "Yes" : "No", x.availability, registryRank[x.id] ?? null,
]);
grid(catalog.getRangeByIndexes(4, 0, payload.feature_catalog.length, 9));
catalog.getRange("A:A").format.columnWidth = 31;
catalog.getRange("B:B").format.columnWidth = 26;
catalog.getRange("C:C").format.columnWidth = 28;
catalog.getRange("D:D").format.columnWidth = 52;
catalog.getRange("E:E").format.columnWidth = 24;
catalog.getRange("F:F").format.columnWidth = 33;
catalog.getRange("G:G").format.columnWidth = 18;
catalog.getRange("H:H").format.columnWidth = 36;
catalog.getRange("I:I").format.columnWidth = 14;
catalog.freezePanes.freezeRows(4);

for (const run of payload.runs_detail) {
  const sheet = workbook.worksheets.add(uniqueSheetName(workbook, `Labels-${run.run}`));
  const status = run.validation.status || (run.labelled_track_count ? "historical_labels" : "awaiting_checklist");
  title(sheet, `Labels — ${run.run}`, `One row per stabilized person track. This is the reusable review sheet for the run; labels and probabilities feed the cumulative predictor after validation.`);
  sheet.getRange("A4:G4").values = [["Track ID", "Frames present", "Reviewed label", "Predicted label", "Dancer probability", "Validation status", "Correction recorded"]];
  header(sheet.getRange("A4:G4"));
  sheet.getRangeByIndexes(4, 0, run.tracks.length, 7).values = run.tracks.map((t) => [
    t.track_id, t.frames_present, t.reviewed_label || "", t.predicted_label, t.dancer_probability,
    status.replaceAll("_", " "), (run.validation.corrected_track_ids || []).includes(t.track_id) ? "Yes" : "",
  ]);
  grid(sheet.getRangeByIndexes(4, 0, run.tracks.length, 7));
  sheet.getRange(`E5:E${4 + run.tracks.length}`).format.numberFormat = "0.0%";
  sheet.getRange("I4:L4").values = [["Registry top 5 used for prediction", "Std. coefficient", "Absolute weight", "Direction"]];
  header(sheet.getRange("I4:L4"));
  const scoringTop5 = run.scoring_top5 || run.run_top5;
  const localRows = scoringTop5.length ? scoringTop5.map((x) => [x.feature_name, x.standardized_coefficient, x.absolute_weight, x.direction]) : [["Awaiting at least two dancer and two non-dancer labels", null, null, null]];
  sheet.getRangeByIndexes(4, 8, localRows.length, 4).values = localRows;
  grid(sheet.getRangeByIndexes(4, 8, localRows.length, 4));
  if (scoringTop5.length) sheet.getRange(`J5:K${4 + localRows.length}`).format.numberFormat = "0.000";
  sheet.getRange("A:A").format.columnWidth = 13;
  sheet.getRange("B:B").format.columnWidth = 16;
  sheet.getRange("C:D").format.columnWidth = 18;
  sheet.getRange("E:E").format.columnWidth = 20;
  sheet.getRange("F:G").format.columnWidth = 18;
  sheet.getRange("I:I").format.columnWidth = 33;
  sheet.getRange("J:K").format.columnWidth = 18;
  sheet.getRange("L:L").format.columnWidth = 22;
  sheet.freezePanes.freezeRows(4);

  for (const track of run.tracks.filter((t) => t.reviewed_label === "dancer")) {
    const detail = workbook.worksheets.add(uniqueSheetName(workbook, `D${track.track_id + 1}-${run.run}`));
    title(detail, `Run ${run.run} — dancer track ${track.track_id + 1}`, `Validated dancer. Model probability: ${(track.dancer_probability * 100).toFixed(1)}%. Top five global predictors are shown with this track's contribution.`);
    detail.getRange("A4:B7").values = [
      ["Frames present", track.frames_present], ["Predicted probability", track.dancer_probability],
      ["Predicted label", track.predicted_label], ["Validated label", track.reviewed_label],
    ];
    detail.getRange("A4:A7").format = { fill: "#DCEAF7", font: { name: font, bold: true, color: navy } };
    grid(detail.getRange("A4:B7")); detail.getRange("B5").format.numberFormat = "0.0%";
    detail.getRange("A9:G9").values = [["Rank", "Predictor", "Category", "Raw value", "Std. value", "Std. coefficient", "Track logit contribution"]];
    header(detail.getRange("A9:G9"));
    detail.getRangeByIndexes(9, 0, track.top_predictors.length, 7).values = track.top_predictors.map((p) => [
      p.rank, p.feature_name, p.category, p.raw_value, p.standardized_value, p.standardized_coefficient, p.logit_contribution,
    ]);
    grid(detail.getRangeByIndexes(9, 0, track.top_predictors.length, 7));
    detail.getRange(`D10:G${9 + track.top_predictors.length}`).format.numberFormat = "0.000";
    detail.getRange("A:G").format.columnWidth = 20;
    detail.getRange("B:B").format.columnWidth = 32;
    detail.getRange("C:C").format.columnWidth = 30;
    detail.freezePanes.freezeRows(9);
  }
}

workbook.recalculate();
const inspection = await workbook.inspect({ kind: "workbook,sheet,table", maxChars: 5000, tableMaxRows: 5, tableMaxCols: 8 });
await fs.writeFile(path.join(runsDir, "dancer_feature_registry_verification.json"), inspection.ndjson || String(inspection));
const preview = await workbook.render({ sheetName: "Registry Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
console.log(outputPath);
