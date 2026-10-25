"""`hermes sessions optimize-storage` runs in the foreground behind the held-store refusal, so it must
report real progress in every phase and never idle between chunks."""

import sqlite3
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import LEGACY_FTS_SQL, SCHEMA_SQL


def _legacy_inline_store(db_path, n=1200):
    raw = sqlite3.connect(str(db_path))
    try:
        raw.executescript(SCHEMA_SQL + LEGACY_FTS_SQL)
        raw.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)", (time.time(),))
        raw.executemany(
            "INSERT INTO messages (session_id, timestamp, role, content) VALUES ('s1', ?, 'user', ?)",
            [(time.time(), f"legacy payload {i} zebra quokka") for i in range(n)],
        )
        raw.commit()
    finally:
        raw.close()


@pytest.fixture
def legacy_db(tmp_path):
    _legacy_inline_store(tmp_path / "state.db")
    db = SessionDB(db_path=tmp_path / "state.db")
    assert db.fts_optimize_available()
    yield db
    db.close()


def test_teardown_reports_which_table_and_how_many_rows(legacy_db):
    events = []
    result = legacy_db.optimize_fts_storage(progress_cb=events.append, vacuum=False)
    assert result["ok"] is True
    teardown = [e for e in events if e["phase"] == "teardown"]
    # The docsize shadow table holds one row per message: its drain must be visible as rows, not a bare label.
    docsize = [e for e in teardown if e.get("table", "").endswith("messages_fts_docsize")]
    assert docsize and docsize[0]["total"] == 1200
    assert docsize[-1]["indexed"] == 1200 and docsize[-1]["percent"] == 100
    count = teardown[0]["table_count"]
    assert count >= 2 and {e["table_index"] for e in teardown} <= set(range(1, count + 1))
    assert legacy_db.search_messages("quokka")


def test_foreground_optimize_never_sleeps_between_chunks(legacy_db, monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    assert legacy_db.optimize_fts_storage(vacuum=False)["ok"] is True
    assert slept == []
