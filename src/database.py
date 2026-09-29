"""SQLite persistence for ElectroDiagnose diagnosis history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data") / "electrodiagnose.db"


def _connect(db_path=DEFAULT_DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    return c


def initialize_database(db_path=DEFAULT_DB_PATH):
    with _connect(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            device TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            priority TEXT NOT NULL,
            summary TEXT,
            findings_json TEXT NOT NULL,
            causes_json TEXT NOT NULL,
            actions_json TEXT NOT NULL,
            limitations_json TEXT NOT NULL,
            review_required INTEGER NOT NULL,
            reasoning_json TEXT NOT NULL DEFAULT '{}',
            evidence_json TEXT NOT NULL DEFAULT '{}'
        )""")
        columns = {row["name"] for row in c.execute("PRAGMA table_info(diagnoses)").fetchall()}
        if "reasoning_json" not in columns:
            c.execute("ALTER TABLE diagnoses ADD COLUMN reasoning_json TEXT NOT NULL DEFAULT '{}'")
        if "evidence_json" not in columns:
            c.execute("ALTER TABLE diagnoses ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '{}'")


def save_diagnosis(diagnosis: dict[str, Any], evidence: dict[str, Any] | None = None, db_path=DEFAULT_DB_PATH) -> int:
    initialize_database(db_path)
    evidence = evidence or {}
    reasoning = evidence.get("reasoning", {})
    stored_evidence = {
        "images": list(evidence.get("images", [])),
        "observations": list(reasoning.get("observations", [])),
        "evidence": list(reasoning.get("evidence", [])),
        "symptoms": str(evidence.get("symptoms", "")),
        "ocr_text": str(evidence.get("ocr_text", "")),
    }
    with _connect(db_path) as c:
        cur = c.execute(
            """INSERT INTO diagnoses
            (timestamp,device,status,confidence,severity,priority,summary,
             findings_json,causes_json,actions_json,limitations_json,
             review_required,reasoning_json,evidence_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                diagnosis.get("device", "unknown"),
                diagnosis.get("status", "INSUFFICIENT_EVIDENCE"),
                float(diagnosis.get("confidence") or 0),
                diagnosis.get("severity", "UNKNOWN"),
                diagnosis.get("priority", "LOW"),
                diagnosis.get("summary", ""),
                json.dumps(diagnosis.get("findings", [])),
                json.dumps(diagnosis.get("possible_causes", [])),
                json.dumps(diagnosis.get("next_actions", [])),
                json.dumps(diagnosis.get("limitations", [])),
                int(bool(diagnosis.get("review_required"))),
                json.dumps(reasoning),
                json.dumps(stored_evidence),
            ),
        )
        return int(cur.lastrowid)


def _decode(row):
    d = dict(row)
    d["findings"] = json.loads(d.pop("findings_json"))
    d["possible_causes"] = json.loads(d.pop("causes_json"))
    d["next_actions"] = json.loads(d.pop("actions_json"))
    d["limitations"] = json.loads(d.pop("limitations_json"))
    d["review_required"] = bool(d["review_required"])
    d["reasoning"] = json.loads(d.pop("reasoning_json", "{}") or "{}")
    d["evidence"] = json.loads(d.pop("evidence_json", "{}") or "{}")
    return d


def list_diagnoses(limit=20, db_path=DEFAULT_DB_PATH):
    initialize_database(db_path)
    limit = max(1, min(int(limit), 100))
    with _connect(db_path) as c:
        rows = c.execute("SELECT * FROM diagnoses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_decode(r) for r in rows]


def get_diagnosis(diagnosis_id: int, db_path=DEFAULT_DB_PATH):
    initialize_database(db_path)
    with _connect(db_path) as c:
        row = c.execute("SELECT * FROM diagnoses WHERE id=?", (int(diagnosis_id),)).fetchone()
    return None if row is None else _decode(row)
