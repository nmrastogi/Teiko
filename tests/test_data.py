"""
tests/test_data.py – Tests for duplicate and bad data handling in load_data.py.

Run with:  pytest tests/test_data.py -v
"""

import csv
import io
import sqlite3
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from load_data import init_db, load_csv, populate_frequencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_conn():
    """Return an in-memory SQLite connection with the schema initialised."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def write_csv(tmp_path, rows, fieldnames=None):
    """Write a list-of-dicts to a temp CSV and return the file path."""
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    p = tmp_path / "test.csv"
    with open(p, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


VALID_ROW = {
    "project":                  "proj1",
    "subject":                  "sub1",
    "condition":                "melanoma",
    "age":                      "42",
    "sex":                      "M",
    "treatment":                "miraclib",
    "response":                 "yes",
    "sample":                   "s001",
    "sample_type":              "PBMC",
    "time_from_treatment_start": "0",
    "b_cell":                   "100",
    "cd8_t_cell":               "200",
    "cd4_t_cell":               "300",
    "nk_cell":                  "150",
    "monocyte":                 "250",
}


def row(**overrides):
    """Return a copy of VALID_ROW with any fields overridden."""
    r = dict(VALID_ROW)
    r.update(overrides)
    return r


# ---------------------------------------------------------------------------
# Duplicate data tests
# ---------------------------------------------------------------------------

class TestDuplicateData:

    def test_duplicate_sample_same_file_ignored(self, tmp_path):
        """Two identical rows → only one inserted (INSERT OR IGNORE)."""
        conn = make_conn()
        csv_path = write_csv(tmp_path, [VALID_ROW, VALID_ROW])
        load_csv(conn, csv_path)
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        assert count == 1

    def test_load_same_csv_twice_idempotent(self, tmp_path):
        """Loading the same CSV a second time does not change row counts."""
        conn = make_conn()
        csv_path = write_csv(tmp_path, [VALID_ROW])
        load_csv(conn, csv_path)
        load_csv(conn, csv_path)
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cell_counts").fetchone()[0] == 1

    def test_duplicate_subject_across_samples(self, tmp_path):
        """Same subject with two different samples → 1 subject, 2 samples."""
        rows = [
            row(sample="s001", time_from_treatment_start="0"),
            row(sample="s002", time_from_treatment_start="7"),
        ]
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, rows))
        assert conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 2

    def test_duplicate_project_not_duplicated(self, tmp_path):
        """Two samples in the same project → project inserted only once."""
        rows = [
            row(sample="s001"),
            row(sample="s002", subject="sub2"),
        ]
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, rows))
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1

    def test_duplicate_sample_different_counts_keeps_first(self, tmp_path):
        """Same sample_id with different cell counts → first wins (INSERT OR IGNORE)."""
        rows = [
            row(sample="s001", b_cell="100"),
            row(sample="s001", b_cell="999"),
        ]
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, rows))
        b = conn.execute("SELECT b_cell FROM cell_counts WHERE sample_id='s001'").fetchone()[0]
        assert b == 100

    def test_populate_frequencies_idempotent(self, tmp_path):
        """Calling populate_frequencies twice does not duplicate rows."""
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [VALID_ROW]))
        populate_frequencies(conn)
        populate_frequencies(conn)
        count = conn.execute("SELECT COUNT(*) FROM cell_frequencies").fetchone()[0]
        assert count == 5  # one row per population


# ---------------------------------------------------------------------------
# Bad data tests
# ---------------------------------------------------------------------------

class TestBadData:

    def test_missing_csv_file_raises(self):
        """Passing a non-existent CSV path raises OSError."""
        conn = make_conn()
        with pytest.raises(OSError):
            load_csv(conn, "/nonexistent/path/file.csv")

    def test_non_integer_cell_count_raises(self, tmp_path):
        """A non-numeric cell count raises ValueError and rolls back."""
        bad = row(b_cell="not_a_number")
        conn = make_conn()
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            load_csv(conn, write_csv(tmp_path, [bad]))
        # rollback: no samples committed
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0

    def test_non_integer_age_stored_as_null(self, tmp_path):
        """A blank age is stored as NULL without raising an error."""
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [row(age="")]))
        age = conn.execute("SELECT age FROM subjects WHERE subject_id='sub1'").fetchone()[0]
        assert age is None

    def test_missing_response_stored_as_null(self, tmp_path):
        """A blank response field is stored as NULL."""
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [row(response="")]))
        resp = conn.execute("SELECT response FROM samples WHERE sample_id='s001'").fetchone()[0]
        assert resp is None

    def test_missing_required_column_raises(self, tmp_path):
        """CSV missing the 'sample' column raises KeyError."""
        bad = dict(VALID_ROW)
        del bad["sample"]
        conn = make_conn()
        with pytest.raises(KeyError):
            load_csv(conn, write_csv(tmp_path, [bad], fieldnames=list(bad.keys())))

    def test_negative_cell_count_inserted(self, tmp_path):
        """Negative cell counts are stored as-is (schema has no CHECK constraint)."""
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [row(b_cell="-5")]))
        b = conn.execute("SELECT b_cell FROM cell_counts WHERE sample_id='s001'").fetchone()[0]
        assert b == -5

    def test_empty_csv_no_rows_inserted(self, tmp_path):
        """An empty CSV (header only) inserts nothing."""
        p = tmp_path / "empty.csv"
        with open(p, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(VALID_ROW.keys())).writeheader()
        conn = make_conn()
        load_csv(conn, str(p))
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0

    def test_frequencies_percentages_sum_to_100(self, tmp_path):
        """After populate_frequencies, percentages for one sample sum to 100."""
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [VALID_ROW]))
        populate_frequencies(conn)
        total_pct = conn.execute(
            "SELECT ROUND(SUM(percentage), 2) FROM cell_frequencies WHERE sample_id='s001'"
        ).fetchone()[0]
        assert total_pct == 100.0

    def test_frequencies_correct_values(self, tmp_path):
        """Percentages are computed correctly from known cell counts."""
        # b=100, cd8=200, cd4=300, nk=150, monocyte=250 → total=1000
        conn = make_conn()
        load_csv(conn, write_csv(tmp_path, [VALID_ROW]))
        populate_frequencies(conn)
        pct = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT population, percentage FROM cell_frequencies WHERE sample_id='s001'"
            ).fetchall()
        }
        assert pct["b_cell"]     == pytest.approx(10.0, abs=0.01)
        assert pct["cd8_t_cell"] == pytest.approx(20.0, abs=0.01)
        assert pct["cd4_t_cell"] == pytest.approx(30.0, abs=0.01)
        assert pct["nk_cell"]    == pytest.approx(15.0, abs=0.01)
        assert pct["monocyte"]   == pytest.approx(25.0, abs=0.01)
