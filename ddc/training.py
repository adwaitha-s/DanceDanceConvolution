"""Persistent next-ten-video cohort, Excel review, LASSO and skeleton LSTM.

Run ``python -m ddc.training --help``. Pose detection is never repeated here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dancer_features import load_labels
from .feature_inventory import collect_feature_inventory
from .feature_registry import load_merges, review_is_current
from .lasso_training import fit_lasso, score
from .skeleton_lstm import middle_sequences, train_lstm
from .training_features import FEATURES, FEATURE_IDS

COHORT = "training_cohort.json"
WORKBOOK = "training_registry.xlsx"


def _write(path, data):
    # Atomic replacement: readers see either complete version, never partial JSON.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def _lock(root):
    from filelock import FileLock
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".training.lock"), timeout=30):
        yield


def arm(runs_dir, target=10):
    root = Path(runs_dir)
    with _lock(root):
        path = root / COHORT
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        state = {"created_at": datetime.now(timezone.utc).isoformat(), "target": target,
                 "existing_runs": sorted(p.name for p in root.iterdir() if p.is_dir()),
                 "runs": [], "status": "collecting", "target_label": "dancer vs non_dancer"}
        _write(path, state)
        return state


def register_run(runs_dir, run_name):
    """Enroll only completed new analyses; reopen/review never consumes a slot."""
    root = Path(runs_dir)
    with _lock(root):
        path = root / COHORT
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if run_name in state["existing_runs"] or run_name in state["runs"]:
            return state
        if len(state["runs"]) >= state["target"]:
            return state
        run = root / run_name
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        if meta.get("frames", 0) <= 0 or not (run / "tracking.jsonl").stat().st_size:
            raise ValueError("Only successful nonempty video runs can be enrolled")
        state["runs"].append(run_name)
        meta.update(training_run=True, training_index=len(state["runs"]))
        _write(run / "meta.json", meta)
        _write(path, state)
        return state


def signature(run_dir, merges=None):
    from .tracking import TRACKING_VERSION
    from .appearance import source_stamp
    digest = hashlib.sha256()
    digest.update(f"tracking-v{TRACKING_VERSION}".encode())
    digest.update(json.dumps(source_stamp(Path(run_dir) / "tracking.jsonl")).encode())
    with (Path(run_dir) / "tracking.jsonl").open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    digest.update(json.dumps(load_merges(Path(run_dir)) if merges is None else merges).encode())
    return digest.hexdigest()


def record_screening(runs_dir, run_name, predictions, merges):
    """Keep precisely the predictions shown in Dancers for later Excel validation."""
    run = Path(runs_dir) / run_name
    labels = [{key: p[key] for key in ("track_id", "frames_present", "predicted_label", "dancer_probability")
               if key in p} for p in predictions]
    _write(run / "dancer_screening.json", {"signature": signature(run, merges),
           "predictions": labels, "generated_at": datetime.now(timezone.utc).isoformat()})


def _discover_runs(root, state):
    """Discover finished new run folders, stopping at the ten-slot cohort limit."""
    missing = [name for name in state["runs"] if not (root / name / "tracking.jsonl").exists()]
    if missing:
        state["missing_runs"] = sorted(set(state.get("missing_runs", [])) | set(missing))
        state["runs"] = [name for name in state["runs"] if name not in missing]
    excluded = set(state["existing_runs"]) | set(state["runs"])
    for run in sorted(root.iterdir(), key=lambda p: p.name):
        if len(state["runs"]) >= state["target"]:
            break
        if not run.is_dir() or run.name in excluded:
            continue
        try:
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            if meta.get("frames", 0) <= 0 or not (run / "tracking.jsonl").stat().st_size:
                continue
        except (OSError, ValueError):
            continue  # An analysis still writing its output is not a completed run.
        state["runs"].append(run.name)
        meta.update(training_run=True, training_index=len(state["runs"]))
        _write(run / "meta.json", meta)
    for index, name in enumerate(state["runs"], 1):
        path = root / name / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("training_index") != index:
            meta["training_index"] = index
            _write(path, meta)


def inputs_stamp(runs_dir):
    """Cheap file metadata used by the app to refresh only after input changes."""
    root = Path(runs_dir)
    if not (root / COHORT).exists():
        return None
    from .tracking import TRACKING_VERSION
    entries = [("tracking_version", TRACKING_VERSION)]
    for run in sorted(root.iterdir(), key=lambda p: p.name):
        if run.is_dir() and (run / "meta.json").exists():
            for name in ("meta.json", "tracking.jsonl", "labels.csv", "merges.json",
                         "dancer_screening.json", "prediction_validation.json",
                         *[p.name for p in run.glob("source.*")]):
                try:
                    stat = (run / name).stat()
                    entries.append((run.name, name, stat.st_size, stat.st_mtime_ns))
                except OSError:
                    pass
    return tuple(entries)


def _feature_rows(run, merges, sig):
    cache = run / "training/features.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("signature") == sig:
            return saved["tracks"]
    rows = collect_feature_inventory(run / "tracking.jsonl", merges)
    for row in rows:
        row["values"] = {fid: (float(row["values"][fid]) if isinstance(row["values"].get(fid),
                          (int, float)) and np.isfinite(row["values"][fid]) else None)
                          for fid in FEATURE_IDS}
    _write(cache, {"signature": sig, "tracks": rows})
    return rows


def _sequences(run, merges, sig):
    folder = run / "training"
    metadata, data = folder / "preprocessing.json", folder / "middle_5s_skeleton.npz"
    if metadata.exists() and data.exists():
        crop = json.loads(metadata.read_text(encoding="utf-8"))
        if crop.get("signature") == sig:
            with np.load(data, allow_pickle=False) as saved:
                tracks = {int(tid): {"sequence": seq, "coverage": float(seq[:, 34:].mean())}
                          for tid, seq in zip(saved["track_ids"], saved["x"])}
            return tracks, crop
    tracks, crop = middle_sequences(run, merges)
    folder.mkdir(exist_ok=True)
    np.savez_compressed(data, x=np.stack([s["sequence"] for s in tracks.values()]) if tracks
                       else np.empty((0, 100, 51), np.float32), track_ids=np.asarray(list(tracks)))
    _write(metadata, {**crop, "signature": sig})
    return tracks, crop


def replace_run(runs_dir, run_name):
    """Keep the original files, freeing a cohort slot for the next successful video."""
    root = Path(runs_dir)
    with _lock(root):
        state = json.loads((root / COHORT).read_text(encoding="utf-8"))
        if run_name not in state["runs"]:
            raise ValueError("Replacement must name a currently enrolled training run")
        state["runs"].remove(run_name)
        state["existing_runs"].append(run_name)
        state.update(status="collecting", lstm_current=False)
        meta_path = root / run_name / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(training_run=False, training_replaced=True)
        _write(meta_path, meta)
        for index, name in enumerate(state["runs"], 1):
            meta_path = root / name / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["training_index"] = index
            _write(meta_path, meta)
        _write(root / COHORT, state)
        return state


def import_reviews(runs_dir, workbook=None):
    """Validate the entire import before writing labels; blank labels stay unreviewed."""
    from openpyxl import load_workbook
    root = Path(runs_dir)
    path = Path(workbook) if workbook else root / WORKBOOK
    if not path.exists():
        return 0
    state = json.loads((root / COHORT).read_text(encoding="utf-8"))
    pending = []
    wb = load_workbook(path, read_only=True, data_only=False)
    try:
        for index, run_name in enumerate(state["runs"], 1):
            sheet_name = f"Run {index:02d}"
            if sheet_name not in wb.sheetnames:
                continue  # A just-enrolled run is not in the preceding export yet.
            ws = wb[sheet_name]
            if ws["B2"].value != run_name or ws["B3"].value != signature(root / run_name):
                # Merging tracks invalidates worksheet ids. Never attach old reviews to new people.
                if workbook and ws["B2"].value is not None:
                    raise ValueError(f"{sheet_name}: stale run or pose signature; download the refreshed workbook")
                continue
            label_path = root / run_name / "labels.csv"
            if label_path.exists() and label_path.stat().st_mtime_ns > path.stat().st_mtime_ns:
                continue  # A newer UI review takes precedence over an older Excel export.
            headers = [ws.cell(6, c).value for c in range(1, 7)]
            if (headers[:3] != ["Track id", "Frames present", "Predicted label"]
                    or headers[3] not in ("LASSO score", "Predictor score")
                    or headers[4] not in ("Manual label", "Validation label")
                    or headers[5] != "Review status"):
                raise ValueError(f"{sheet_name}: review column headings have changed")
            valid_ids = {r["track_id"] for r in collect_feature_inventory(
                root / run_name / "tracking.jsonl", load_merges(root / run_name))}
            labels, seen = {}, set()
            for row in ws.iter_rows(min_row=7, values_only=True):
                if row[0] is None:
                    break
                tid, label = row[0], row[4]
                if not isinstance(tid, int) or tid not in valid_ids or tid in seen:
                    raise ValueError(f"{sheet_name}: unknown or duplicate track id {tid}")
                seen.add(tid)
                if label in (None, ""):
                    continue
                if label not in ("dancer", "non_dancer"):
                    raise ValueError(f"{sheet_name}, track {tid}: label must be dancer or non_dancer")
                labels[tid] = int(label == "dancer")
            if seen != valid_ids:
                raise ValueError(f"{sheet_name}: review table is missing track rows")
            if seen:
                pending.append((run_name, labels))
    finally:
        wb.close()
    for run_name, labels in pending:
        run = root / run_name
        with (run / "labels.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "label"])
            writer.writerows((tid, "dancer" if label else "non_dancer") for tid, label in sorted(labels.items()))
        _write(run / "training_review.json", {"signature": signature(run),
               "reviewed_at": datetime.now(timezone.utc).isoformat(), "source": "Excel"})
    return sum(len(labels) for _, labels in pending)


def export_workbook(root, payload):
    """Use the bundled artifact tool when present, or configured Node/package runtime."""
    root = Path(root)
    _write(root / "training_workbook_payload.json", payload)
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = os.environ.get("DDC_NODE") or (str(bundled / "bin/node.exe")
             if (bundled / "bin/node.exe").exists() else shutil.which("node"))
    modules = os.environ.get("DDC_ARTIFACT_MODULE_ROOT") or str(bundled / "node_modules")
    if not node:
        raise RuntimeError("Excel export requires Node.js and @oai/artifact-tool")
    builder = Path(__file__).resolve().parent.parent / "scripts/build_training_workbook.mjs"
    done = subprocess.run([node, str(builder), str(root.resolve()), modules],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    if done.returncode:
        raise RuntimeError(f"Excel export failed: {done.stderr[-1500:]}")
    return root / WORKBOOK


def refresh(runs_dir, train=True, epochs=60, import_excel=True, export=True, review_workbook=None):
    root = Path(runs_dir)
    with _lock(root):
        path = root / COHORT
        if not path.exists():
            raise ValueError("Arm the next-ten-run cohort first")
        state = json.loads(path.read_text(encoding="utf-8"))
        _discover_runs(root, state)
        _write(path, state)
        if import_excel:
            import_reviews(root, review_workbook)
        runs, records, problems, samples = [], [], [], []
        for index, name in enumerate(state["runs"], 1):
            run = root / name
            merges = load_merges(run)
            sig = signature(run)
            try:
                rows = _feature_rows(run, merges, sig)
            except ValueError as exc:
                rows = []
                problems.append(f"{name}: {exc}")
            labels = load_labels(run / "labels.csv") if (run / "labels.csv").exists() else {}
            if not review_is_current(run):
                labels = {}
                problems.append(f"{name}: tracking identities changed; labels must be reviewed again")
            # Store only finite values in the JSON/Excel boundary.
            for row in rows:
                row["manual_label"] = ("dancer" if labels[row["track_id"]] else "non_dancer") \
                    if row["track_id"] in labels else ""
                if row["track_id"] in labels:
                    records.append({**row, "run": name, "label": labels[row["track_id"]]})
            if not rows or any(r["track_id"] not in labels for r in rows):
                problems.append(f"{name}: review every track's dancer/non_dancer label")
            try:
                sequences, crop = _sequences(run, merges, sig)
                for row in rows:
                    tid = row["track_id"]
                    if tid not in sequences:
                        problems.append(f"{name}, track {tid}: less than 50% middle-window joint coverage")
                    elif tid in labels:
                        samples.append({**sequences[tid], "run": name, "track_id": tid,
                                        "label": labels[tid], "crop": crop})
            except ValueError as exc:
                crop = {"error": str(exc)}
                problems.append(f"{name}: {exc}")
            runs.append({"index": index, "run": name, "signature": sig, "tracks": rows, "crop": crop})
        model = None
        try:
            model = fit_lasso(records)
        except ValueError as exc:
            problems.append(str(exc))
        # Predictions are fit without this run, including alpha/feature selection.
        for run in runs:
            try:
                heldout = fit_lasso([r for r in records if r["run"] != run["run"]])
            except ValueError:
                heldout = None
            for row in run["tracks"]:
                row["prediction"] = score(row, heldout) if heldout else None
                row["prediction_features"] = heldout["feature_ids"] if heldout else []
                row["predictor_method"] = "LASSO linear score" if heldout else "Unavailable"
                row["review_action"] = ""
            folder = root / run["run"]
            screening_path = folder / "dancer_screening.json"
            screening = json.loads(screening_path.read_text()) if screening_path.exists() else {}
            saved_path = folder / "prediction_validation.json"
            saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
            review_path = folder / "training_review.json"
            review = json.loads(review_path.read_text()) if review_path.exists() else {}
            if screening.get("signature") == run["signature"]:
                displayed = {int(p["track_id"]): p for p in screening["predictions"]}
                for row in run["tracks"]:
                    shown = displayed.get(row["track_id"])
                    if shown:
                        row["prediction"] = {"predicted_label": shown["predicted_label"],
                                             "score": shown["dancer_probability"]}
                        row["predictor_method"] = "Dancers tab probability"
            for row in run["tracks"]:
                key = str(row["track_id"])
                # A later model refit must not rewrite the prediction that was reviewed.
                if (row["manual_label"] and review.get("signature") == run["signature"]
                        and saved.get("predictor_labels", {}).get(key)):
                    row["prediction"] = {"predicted_label": saved["predictor_labels"][key],
                        "score": saved.get("predictor_probabilities", {}).get(key)}
                    row["predictor_method"] = "Dancers tab probability"
                row["review_action"] = saved.get("review_action", "Saved review") if row["manual_label"] else ""
                predicted = (row["prediction"] or {}).get("predicted_label")
                row["validation_status"] = ("Unreviewed" if not row["manual_label"] else
                    "Reviewed without prediction" if not predicted else
                    "Validated correct" if predicted == row["manual_label"] else "Corrected")
        run_signatures = {r["run"]: r["signature"] for r in runs}
        fingerprint = hashlib.sha256(json.dumps({
                   "samples": [(s["run"], s["track_id"], s["label"], run_signatures[s["run"]]) for s in samples],
                   "model": model, "epochs": epochs}, sort_keys=True).encode()).hexdigest()
        report_path = root / "training/lstm_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        # Unreviewed/new clips do not stop updates from the reviewed usable clips.
        ready = model is not None and {s["label"] for s in samples} == {0, 1}
        complete = len(runs) == state["target"] and not problems
        training_failed = False
        if ready and train and (not report or report.get("fingerprint") != fingerprint):
            state.update(status="training", lstm_current=False)
            _write(path, state)
            try:
                report = train_lstm(samples, root / "training", model["feature_ids"], fingerprint, epochs)
            except (ImportError, OSError, RuntimeError) as exc:
                problems.append(f"LSTM training failed: {exc}")
                training_failed = True
                ready = False
        current = bool(ready and report and report.get("fingerprint") == fingerprint)
        status = ("trained" if complete else "rolling_trained") if current else ("training_failed" if training_failed else "ready_to_train" if ready else
                    "collecting" if len(runs) < state["target"] else "awaiting_review_or_data")
        state.update(status=status, problems=problems, reviewed_tracks=len(records),
                     selected_features=model["feature_ids"] if model else [],
                     lstm_current=current, updated_at=datetime.now(timezone.utc).isoformat())
        state["lstm_runs"] = report["runs"] if current else []
        state["lstm_samples"] = len(samples)
        _write(path, state)
        if model:
            _write(root / "training/lasso_model.json", {**model, "fingerprint": fingerprint})
        payload = {"state": state, "features": [s.__dict__ for s in FEATURES],
                   "model": model, "runs": runs, "lstm": report if current else None}
        _write(root / "training_status.json", {"state": state, "lstm": payload["lstm"]})
        if export:
            export_workbook(root, payload)
        return state


def refresh_after_review(runs_dir):
    """Application hook: a workbook/model error must not discard a saved review."""
    if not (Path(runs_dir) / COHORT).exists():
        return ""
    try:
        state = refresh(runs_dir, import_excel=False)
        return f" Training cohort: {len(state['runs'])}/{state['target']} videos, {state['status']}."
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        return f" Training refresh needs attention: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["arm", "refresh", "status", "import-review", "replace-run"])
    parser.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("DDC_RUNS_DIR", "runs")))
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--run", help="Enrolled run to retain but replace in the training cohort")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    if args.command == "arm":
        arm(args.runs_dir)
        result = refresh(args.runs_dir, train=False)
    elif args.command == "status":
        result = json.loads((args.runs_dir / COHORT).read_text())
    else:
        if args.command == "replace-run":
            if not args.run:
                parser.error("replace-run requires --run")
            replace_run(args.runs_dir, args.run)
        if args.command == "import-review":
            with _lock(args.runs_dir):
                import_reviews(args.runs_dir, args.workbook)
        result = refresh(args.runs_dir, train=not args.no_train, epochs=args.epochs,
                         import_excel=args.command != "import-review")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
