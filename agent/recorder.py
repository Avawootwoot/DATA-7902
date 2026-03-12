import json
import sqlite3
from typing import Any, Dict, Optional


class TranscriptRecorder:
    def __init__(self, db_path: str = "dialogue.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT,
                    focus TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    raw_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_state (
                    session_id TEXT PRIMARY KEY,
                    facts_json TEXT NOT NULL,
                    timeline_json TEXT NOT NULL,
                    open_threads_json TEXT NOT NULL,
                    covered_topics_json TEXT NOT NULL,
                    flags_json TEXT NOT NULL,
                    completed INTEGER NOT NULL,
                    last_intent TEXT,
                    last_focus TEXT
                )
                """
            )

            conn.commit()

    def create_session(self, session_id: str, persona_id: str, status: str, started_at: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, persona_id, status, started_at) VALUES (?, ?, ?, ?)",
                (session_id, persona_id, status, started_at),
            )
            conn.commit()

    def save_turn(
        self,
        session_id: str,
        turn_idx: int,
        role: str,
        content: str,
        intent: Optional[str] = None,
        focus: Optional[str] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns (session_id, turn_idx, role, content, intent, focus)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, turn_idx, role, content, intent, focus),
            )
            conn.commit()

    def save_extraction(self, session_id: str, turn_idx: int, extracted: Dict[str, Any]):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO extractions (session_id, turn_idx, raw_json) VALUES (?, ?, ?)",
                (session_id, turn_idx, json.dumps(extracted, ensure_ascii=False)),
            )
            conn.commit()

    def save_state(self, state):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO planner_state
                (session_id, facts_json, timeline_json, open_threads_json, covered_topics_json,
                 flags_json, completed, last_intent, last_focus)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    json.dumps(state.facts, ensure_ascii=False),
                    json.dumps(state.timeline, ensure_ascii=False),
                    json.dumps(state.open_threads, ensure_ascii=False),
                    json.dumps(state.covered_topics, ensure_ascii=False),
                    json.dumps(state.flags, ensure_ascii=False),
                    int(state.completed),
                    state.last_intent,
                    state.last_focus,
                ),
            )
            conn.commit()

    def close_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ? WHERE session_id = ?",
                ("completed", session_id),
            )
            conn.commit()