"""`hermes sessions optimize-storage` runs in the foreground behind the held-store refusal, so it must
report real progress in every phase, never idle between chunks, and end (not spin) when a chunk
cannot succeed."""

import itertools
import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import LEGACY_FTS_SQL, SCHEMA_SQL

GAP_AFTER, GAP_IDS = 600, 100_000  # a pruned store: ids jump after the first 600 messages


def _legacy_inline_store(db_path, n=1200):
    ids = [i + 1 + (GAP_IDS if i >= GAP_AFTER else 0) for i in range(n)]
    raw = sqlite3.connect(str(db_path))
    try:
        raw.executescript(SCHEMA_SQL + LEGACY_FTS_SQL)
        raw.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)", (time.time(),))
        raw.executemany(
            "INSERT INTO messages (id, session_id, timestamp, role, content) VALUES (?, 's1', ?, 'user', ?)",
            [(mid, time.time(), f"legacy payload {mid} zebra quokka") for mid in ids],
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


@pytest.mark.parametrize("vacuum_fails", [False, True], ids=["vacuum-ok", "vacuum-fails"])
def test_progress_reports_teardown_rows_and_vacuum_liveness(legacy_db, monkeypatch, vacuum_fails):
    monkeypatch.setattr(SessionDB, "_VACUUM_TICK_S", 0.01)
    events, in_vacuum, ticked_in_vacuum = [], threading.Event(), threading.Event()

    def progress(info):
        events.append(info)
        if info["phase"] == "vacuum" and in_vacuum.is_set():
            ticked_in_vacuum.set()

    # Hold the VACUUM statement (SQLite's own progress hook) until a tick arrives while it runs.
    vacuum_started = []
    legacy_db._conn.set_trace_callback(lambda sql: vacuum_started.append(sql) if sql.strip() == "VACUUM" else None)

    def hold_vacuum():
        if not vacuum_started or in_vacuum.is_set():
            return 0
        in_vacuum.set()
        ticked_in_vacuum.wait(10)
        return 1 if vacuum_fails else 0  # non-zero interrupts the VACUUM

    legacy_db._conn.set_progress_handler(hold_vacuum, 100)
    result = legacy_db.optimize_fts_storage(progress_cb=progress, vacuum=True)

    assert result == {"ok": True, "vacuumed": not vacuum_fails}
    assert ticked_in_vacuum.is_set()
    # The ticker is stopped and joined on success and on failure: nothing can report after return.
    assert not [t for t in threading.enumerate() if t.name == "optimize-vacuum-ticker"]
    assert events[-1]["phase"] == "done"
    vacuum = [e for e in events if e["phase"] == "vacuum"]
    assert vacuum[0]["size_bytes"] >= vacuum[0]["estimated_bytes"] > 0
    teardown = [e for e in events if e["phase"] == "teardown"]
    # The docsize shadow table holds one row per message: its drain must be visible as rows, not a bare label.
    docsize = [e for e in teardown if e["table"].endswith("messages_fts_docsize")]
    assert docsize[0]["total"] == 1200 and docsize[0]["indexed"] == 0
    assert docsize[-1]["indexed"] == 1200 and docsize[-1]["percent"] == 100
    count = teardown[0]["table_count"]
    assert count >= 2 and {e["table_index"] for e in teardown} <= set(range(1, count + 1))
    assert legacy_db.search_messages("quokka")


def test_foreground_optimize_never_sleeps_and_chunks_by_rows_across_id_gaps(legacy_db, monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_CHUNK_MIN_ROWS", 500)
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_CHUNK_MAX_ROWS", 500)
    markers, real_step = [0], legacy_db.fts_rebuild_step

    def observed_step():
        more = real_step()
        status = legacy_db.fts_rebuild_status()
        markers.append(status["indexed"] if status else GAP_IDS + 1200)
        return more

    legacy_db.fts_rebuild_step = observed_step
    assert legacy_db.optimize_fts_storage(vacuum=False)["ok"] is True
    assert slept == []
    with legacy_db._read_ctx() as conn:
        windows = [conn.execute("SELECT COUNT(*) FROM messages WHERE id > ? AND id <= ?", (lo, hi)).fetchone()[0]
                   for lo, hi in itertools.pairwise(markers)]
    # 500-row chunks: the 100k-id gap costs no extra chunk and never widens one past 500 messages.
    assert windows == [500, 500, 200]


def test_chunk_that_keeps_failing_ends_the_run_and_a_rerun_resumes(legacy_db, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # the retry backoff
    calls, real_step = [], legacy_db.fts_rebuild_step

    def disk_fills_during_backfill():
        if not calls:
            pages = legacy_db._conn.execute("PRAGMA page_count").fetchone()[0]
            legacy_db._conn.execute(f"PRAGMA max_page_count = {pages + 5}")
        calls.append(1)
        assert len(calls) < 50, "optimize kept retrying a chunk that can never succeed"
        return real_step()

    legacy_db.fts_rebuild_step = disk_fills_during_backfill
    with pytest.raises(sqlite3.OperationalError, match="full"):
        legacy_db.optimize_fts_storage(vacuum=False)
    legacy_db._conn.execute("PRAGMA max_page_count = 1073741823")
    assert legacy_db.optimize_fts_storage(vacuum=False)["ok"] is True
    assert len(legacy_db.search_messages("quokka", limit=2000)) == 1200
