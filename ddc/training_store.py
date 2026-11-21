"""Transactional training registry. Excel and JSON are rebuildable projections.

Raw videos, tracking streams and sequence arrays remain in run folders. SQLite
keeps cohort state, feature caches, reviewed labels, versioned models/checkpoints
and immutable prediction snapshots. SQLite uses Python's built-in driver.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


DATABASE = "lasso_lstm_training.sqlite3"
LEGACY_DATABASE = "training.sqlite3"
SCHEMA_VERSION = 1


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def now():
    return datetime.now(timezone.utc).isoformat()


class TrainingStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / DATABASE
        # Serialize the one-time schema/migration check, including status readers.
        from filelock import FileLock
        with FileLock(str(self.root / ".training-db-init.lock"), timeout=30):
            legacy = self.root / LEGACY_DATABASE
            if not self.path.exists() and legacy.exists():
                legacy.replace(self.path)
            self._initialize()
            self._migrate_legacy_snapshot()

    def _migrate_legacy_snapshot(self):
        """Retain the last exported weights before the first database refresh."""
        with self.connect() as db:
            if self._get(db, "latest_snapshot") is not None or self._get(db, "cohort") is None:
                return
        path = self.root / "training_workbook_payload.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("lstm") or {}
        model_path = self.root / "training/lasso_model.json"
        model = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}
        fingerprint = (report.get("fingerprint") or model.get("fingerprint")
                       or hashlib.sha256(encode(payload.get("model")).encode()).hexdigest())
        checkpoint_path = self.root / "training/feature_lstm.pt"
        checkpoint = checkpoint_path.read_bytes() if report and checkpoint_path.exists() else None
        # The cohort can contain videos enrolled after this older Excel export.
        self.save_snapshot(payload, fingerprint, checkpoint, update_state=False)

    def _initialize(self):
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"Training database schema {version} is newer than this application")
            if version == 0:
                db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE feature_cache (
                        run TEXT NOT NULL, signature TEXT NOT NULL, tracks_json TEXT NOT NULL,
                        PRIMARY KEY (run, signature));
                    CREATE TABLE review_batches (
                        id INTEGER PRIMARY KEY, run TEXT NOT NULL, signature TEXT NOT NULL,
                        digest TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL,
                        UNIQUE (run, signature, digest));
                    CREATE TABLE labels (
                        batch_id INTEGER NOT NULL REFERENCES review_batches(id),
                        track_id INTEGER NOT NULL, label INTEGER NOT NULL CHECK(label IN (0,1)),
                        PRIMARY KEY (batch_id, track_id));
                    CREATE TABLE current_reviews (
                        run TEXT NOT NULL, signature TEXT NOT NULL,
                        batch_id INTEGER NOT NULL REFERENCES review_batches(id),
                        PRIMARY KEY (run, signature));
                    CREATE TABLE model_versions (
                        fingerprint TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                        lasso_json TEXT NOT NULL, lstm_json TEXT, checkpoint BLOB);
                    CREATE TABLE feature_weights (
                        fingerprint TEXT NOT NULL REFERENCES model_versions(fingerprint),
                        rank INTEGER NOT NULL, feature_id TEXT NOT NULL,
                        ranking_weight REAL NOT NULL, coefficient REAL NOT NULL,
                        median REAL NOT NULL, mean REAL NOT NULL, std REAL NOT NULL CHECK(std > 0),
                        PRIMARY KEY (fingerprint, feature_id), UNIQUE(fingerprint, rank));
                    CREATE TABLE snapshots (
                        id INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL, fingerprint TEXT,
                        payload_json TEXT NOT NULL);
                    CREATE TABLE tracks (
                        snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
                        run TEXT NOT NULL, signature TEXT NOT NULL, track_id INTEGER NOT NULL,
                        frames_present INTEGER NOT NULL, manual_label TEXT
                            CHECK(manual_label IN ('dancer','non_dancer')),
                        PRIMARY KEY(snapshot_id, run, track_id));
                    CREATE TABLE feature_values (
                        snapshot_id INTEGER NOT NULL, run TEXT NOT NULL, track_id INTEGER NOT NULL,
                        feature_id TEXT NOT NULL, value REAL,
                        PRIMARY KEY(snapshot_id, run, track_id, feature_id),
                        FOREIGN KEY(snapshot_id, run, track_id) REFERENCES tracks);
                    CREATE TABLE predictions (
                        snapshot_id INTEGER NOT NULL, run TEXT NOT NULL, track_id INTEGER NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('reviewed_screening','lstm')),
                        predicted_label TEXT CHECK(predicted_label IN ('dancer','non_dancer')),
                        score REAL CHECK(kind != 'lstm' OR score BETWEEN 0 AND 1), method TEXT NOT NULL,
                        fingerprint TEXT, in_training INTEGER CHECK(in_training IN (0,1)),
                        PRIMARY KEY(snapshot_id, run, track_id, kind),
                        FOREIGN KEY(snapshot_id, run, track_id) REFERENCES tracks);
                    CREATE INDEX prediction_history ON predictions(run, track_id, kind);
                    CREATE VIEW current_predictions AS SELECT p.* FROM predictions p
                        WHERE snapshot_id = (SELECT CAST(value AS INTEGER) FROM settings
                                             WHERE key = 'latest_snapshot');
                    CREATE VIEW current_weights AS SELECT w.* FROM feature_weights w
                        JOIN snapshots s ON s.fingerprint = w.fingerprint
                        WHERE s.id = (SELECT CAST(value AS INTEGER) FROM settings
                                      WHERE key = 'latest_snapshot');
                    PRAGMA user_version = 1;
                    COMMIT;
                """)
            # One-time, non-destructive upgrade; thereafter DB state wins over JSON.
            if self._get(db, "cohort") is None:
                legacy = self.root / "training_cohort.json"
                if legacy.exists():
                    self._set(db, "cohort", json.loads(legacy.read_text(encoding="utf-8")))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _get(db, key):
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _set(db, key, value):
        db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, encode(value)))

    def state(self):
        with self.connect() as db:
            return self._get(db, "cohort")

    def save_state(self, state):
        with self.connect() as db:
            self._set(db, "cohort", state)

    def features(self, run, signature, tracks=None):
        with self.connect() as db:
            if tracks is not None:
                db.execute("INSERT OR REPLACE INTO feature_cache VALUES (?,?,?)",
                           (run, signature, encode(tracks)))
                return tracks
            row = db.execute("SELECT tracks_json FROM feature_cache WHERE run=? AND signature=?",
                             (run, signature)).fetchone()
            return json.loads(row[0]) if row else None

    def reviews(self, run, signature, labels=None, source="run files"):
        """Import explicit review inputs; missing CSVs can recover database labels.

        Signature keys prevent labels being attached to different merged identities.
        Empty dict explicitly clears reviews; None reads without changing them.
        """
        with self.connect() as db:
            return self._reviews(db, run, signature, labels, source)

    def _reviews(self, db, run, signature, labels=None, source="run files"):
        if labels is not None:
            digest = hashlib.sha256(encode(labels).encode()).hexdigest()
            db.execute("INSERT INTO review_batches(run,signature,digest,source,created_at) "
                       "VALUES (?,?,?,?,?) ON CONFLICT(run,signature,digest) DO NOTHING",
                       (run, signature, digest, source, now()))
            batch = db.execute("SELECT id FROM review_batches WHERE run=? AND signature=? AND digest=?",
                               (run, signature, digest)).fetchone()[0]
            db.executemany("INSERT INTO labels VALUES (?,?,?) ON CONFLICT(batch_id,track_id) DO NOTHING",
                           [(batch, tid, label) for tid, label in labels.items()])
            db.execute("INSERT INTO current_reviews VALUES (?,?,?) ON CONFLICT(run,signature) "
                       "DO UPDATE SET batch_id=excluded.batch_id", (run, signature, batch))
        return {row[0]: row[1] for row in db.execute(
            "SELECT l.track_id,l.label FROM labels l JOIN current_reviews r ON r.batch_id=l.batch_id "
            "WHERE r.run=? AND r.signature=?", (run, signature))}

    def ingest_reviews(self, run, signature, labels, file_digest):
        """Only changed CSV inputs may supersede the authoritative database review."""
        with self.connect() as db:
            key = f"label_file:{run}:{signature}"
            if self._get(db, key) != file_digest:
                self._reviews(db, run, signature, labels)
                self._set(db, key, file_digest)
            return self._reviews(db, run, signature)

    def import_reviews(self, batches):
        """Commit a whole validated workbook before writing compatibility CSVs.

        Remember the prior file digest so an interrupted CSV export cannot undo
        a newer database review on the next refresh.
        """
        with self.connect() as db:
            for run, signature, labels, prior_file_digest in batches:
                self._reviews(db, run, signature, labels, source="Excel")
                self._set(db, f"label_file:{run}:{signature}", prior_file_digest)

    def model(self, fingerprint):
        with self.connect() as db:
            row = db.execute("SELECT lstm_json,checkpoint FROM model_versions WHERE fingerprint=?",
                             (fingerprint,)).fetchone()
            return (json.loads(row[0]) if row[0] else None, row[1]) if row else (None, None)

    def save_snapshot(self, payload, fingerprint, checkpoint=None, update_state=True):
        """Commit state, model, weights, values and predictions together or not at all."""
        stable = json.loads(encode(payload))
        stable["state"].pop("updated_at", None)
        stable["state"].pop("excel_sync", None)
        digest = hashlib.sha256(encode(stable).encode()).hexdigest()
        model, report = payload.get("model"), payload.get("lstm")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if model:
                db.execute("INSERT INTO model_versions VALUES (?,?,?,?,?) ON CONFLICT(fingerprint) "
                           "DO UPDATE SET lstm_json=COALESCE(excluded.lstm_json,lstm_json), "
                           "checkpoint=COALESCE(excluded.checkpoint,checkpoint)",
                           (fingerprint, now(), encode(model), encode(report) if report else None, checkpoint))
                db.executemany("INSERT INTO feature_weights VALUES (?,?,?,?,?,?,?,?) "
                               "ON CONFLICT(fingerprint,feature_id) DO NOTHING",
                               [(fingerprint, i+1, fid, model["ranking_coefficients"][i],
                                 model["coefficient"][i], model["medians"][i], model["mean"][i], model["std"][i])
                                for i, fid in enumerate(model["feature_ids"])])
            old = db.execute("SELECT id FROM snapshots WHERE digest=?", (digest,)).fetchone()
            if old:
                snapshot = old[0]
            else:
                snapshot = db.execute("INSERT INTO snapshots(digest,created_at,fingerprint,payload_json) "
                                      "VALUES (?,?,?,?)", (digest, now(), fingerprint if model else None,
                                                          encode(payload))).lastrowid
                for run in payload["runs"]:
                    for track in run["tracks"]:
                        key = (snapshot, run["run"], track["track_id"])
                        db.execute("INSERT INTO tracks VALUES (?,?,?,?,?,?)",
                                   (snapshot, run["run"], run["signature"], track["track_id"],
                                    track["frames_present"], track.get("manual_label") or None))
                        db.executemany("INSERT INTO feature_values VALUES (?,?,?,?,?)",
                                       [(*key, fid, value) for fid, value in track["values"].items()])
                        p = track.get("prediction") or {}
                        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                                   (*key, "reviewed_screening", p.get("predicted_label"), p.get("score"),
                                    track.get("predictor_method", "Unavailable"), None, None))
                        p = track.get("lstm_prediction") or {}
                        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                                   (*key, "lstm", p.get("predicted_label"), p.get("probability"),
                                    track.get("lstm_status", "Unavailable"), p.get("fingerprint"),
                                    int(p["in_training"]) if p else None))
            if update_state:
                self._set(db, "cohort", payload["state"])
            self._set(db, "latest_snapshot", snapshot)
            self._set(db, "excel_sync", {"status": "pending", "snapshot": snapshot})
        return snapshot

    def payload(self):
        with self.connect() as db:
            db.execute("BEGIN")  # Read state, history and predictions from one consistent version.
            row = db.execute("SELECT payload_json FROM snapshots WHERE id=?",
                             (self._get(db, "latest_snapshot"),)).fetchone()
            if not row:
                raise ValueError("Refresh training before exporting the database")
            result = json.loads(row[0])
            state = self._get(db, "cohort")
            if state["runs"] != result["state"]["runs"]:
                raise ValueError("Cohort membership changed; refresh before exporting")
            result["state"] = state
            result["database_snapshot"] = self._get(db, "latest_snapshot")
            result["weight_history"] = [dict(row) for row in db.execute(
                "SELECT m.created_at,w.* FROM feature_weights w JOIN model_versions m "
                "ON w.fingerprint=m.fingerprint ORDER BY m.created_at,w.rank")]
            return result

    def sync_status(self, error=None):
        with self.connect() as db:
            if error is not None:
                value = {"status": "pending" if error else "synced", "error": str(error),
                         "snapshot": self._get(db, "latest_snapshot"), "attempted_at": now()}
                self._set(db, "excel_sync", value)
            return self._get(db, "excel_sync")

    def backup(self, destination):
        """Use SQLite's backup API, including committed data from a live connection."""
        destination = Path(destination).resolve()
        if destination == self.path.resolve() or destination.exists():
            raise ValueError("Choose a new backup file, different from the live database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the new path exclusively so two backups cannot overwrite one another.
        destination.open("xb").close()
        with self.connect() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
        return destination


def is_armed(root):
    root = Path(root)
    if (root / DATABASE).exists() or (root / LEGACY_DATABASE).exists():
        return TrainingStore(root).state() is not None
    return (root / "training_cohort.json").exists()
