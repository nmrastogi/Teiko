"""
load_data.py – Part 1: initialise the SQLite database and load cell-count.csv.

Schema rationale
----------------
Four normalised tables mirror the natural hierarchy of clinical data:

  projects  →  subjects  →  samples  →  cell_counts

- projects: one row per trial/cohort; adding new projects never touches
  other tables.
- subjects: demographics (condition, age, sex) stored once per patient,
  not repeated on every sample row.  At scale this avoids data-update
  anomalies when a field is corrected.
- samples: one row per biological sample with collection-time metadata
  (treatment, response, sample_type, time_from_treatment_start).  A subject
  may have multiple samples over time.
- cell_counts: the five measured populations keyed by sample_id.  Separating
  counts from metadata keeps each table focused and makes it straightforward
  to add new assay types (e.g. cytokine panels) without altering the samples
  table.

Indexes are created on the foreign-key columns most likely to appear in WHERE
clauses (condition, treatment, response, sample_type, time_from_treatment_start)
so analytical queries stay fast even with millions of rows.
"""

import csv
import logging
import os
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = "teiko.db"
CSV_PATH = "cell-count.csv"

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id  TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    condition   TEXT NOT NULL,
    age         INTEGER,
    sex         TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id                   TEXT PRIMARY KEY,
    subject_id                  TEXT NOT NULL,
    treatment                   TEXT,
    response                    TEXT,
    sample_type                 TEXT,
    time_from_treatment_start   INTEGER,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE IF NOT EXISTS cell_counts (
    sample_id  TEXT PRIMARY KEY,
    b_cell     INTEGER NOT NULL,
    cd8_t_cell INTEGER NOT NULL,
    cd4_t_cell INTEGER NOT NULL,
    nk_cell    INTEGER NOT NULL,
    monocyte   INTEGER NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);

CREATE TABLE IF NOT EXISTS cell_frequencies (
    sample_id   TEXT NOT NULL,
    population  TEXT NOT NULL,
    count       INTEGER NOT NULL,
    total_count INTEGER NOT NULL,
    percentage  REAL NOT NULL,
    PRIMARY KEY (sample_id, population),
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);

CREATE INDEX IF NOT EXISTS idx_subjects_condition  ON subjects(condition);
CREATE INDEX IF NOT EXISTS idx_subjects_project    ON subjects(project_id);
CREATE INDEX IF NOT EXISTS idx_samples_treatment   ON samples(treatment);
CREATE INDEX IF NOT EXISTS idx_samples_response    ON samples(response);
CREATE INDEX IF NOT EXISTS idx_samples_type        ON samples(sample_type);
CREATE INDEX IF NOT EXISTS idx_samples_time        ON samples(time_from_treatment_start);
CREATE INDEX IF NOT EXISTS idx_freq_population     ON cell_frequencies(population);
CREATE INDEX IF NOT EXISTS idx_freq_sample         ON cell_frequencies(sample_id);
"""


def init_db(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(DDL)
        conn.commit()
    except sqlite3.DatabaseError as e:
        log.error("Failed to initialise schema: %s", e)
        raise


def load_csv(conn: sqlite3.Connection, csv_path: str = CSV_PATH) -> None:
    try:
        fh = open(csv_path, newline="")
    except OSError as e:
        log.error("Cannot open CSV file '%s': %s", csv_path, e)
        raise

    cursor = conn.cursor()
    inserted = {"projects": 0, "subjects": 0, "samples": 0, "cell_counts": 0}

    try:
        with fh:
            for row in csv.DictReader(fh):
                try:
                    # projects
                    cursor.execute(
                        "INSERT OR IGNORE INTO projects (project_id) VALUES (?)",
                        (row["project"],),
                    )
                    if cursor.rowcount:
                        inserted["projects"] += 1

                    # subjects
                    age = int(row["age"]) if row.get("age") else None
                    cursor.execute(
                        """INSERT OR IGNORE INTO subjects
                           (subject_id, project_id, condition, age, sex)
                           VALUES (?, ?, ?, ?, ?)""",
                        (row["subject"], row["project"], row["condition"], age, row["sex"]),
                    )
                    if cursor.rowcount:
                        inserted["subjects"] += 1

                    # samples
                    response = row["response"] if row.get("response") else None
                    cursor.execute(
                        """INSERT OR IGNORE INTO samples
                           (sample_id, subject_id, treatment, response,
                            sample_type, time_from_treatment_start)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            row["sample"],
                            row["subject"],
                            row["treatment"],
                            response,
                            row["sample_type"],
                            int(row["time_from_treatment_start"]),
                        ),
                    )
                    if cursor.rowcount:
                        inserted["samples"] += 1

                    # cell_counts
                    cursor.execute(
                        """INSERT OR IGNORE INTO cell_counts
                           (sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            row["sample"],
                            int(row["b_cell"]),
                            int(row["cd8_t_cell"]),
                            int(row["cd4_t_cell"]),
                            int(row["nk_cell"]),
                            int(row["monocyte"]),
                        ),
                    )
                    if cursor.rowcount:
                        inserted["cell_counts"] += 1

                except (sqlite3.DatabaseError, KeyError, ValueError) as e:
                    log.error("Failed to insert row (sample=%s): %s", row.get("sample", "?"), e)
                    raise

        conn.commit()
        for table, n in inserted.items():
            log.info("  %s: %d rows inserted", table, n)

    except (sqlite3.DatabaseError, KeyError, ValueError) as e:
        log.error("CSV load failed, rolling back: %s", e)
        conn.rollback()
        raise


def populate_frequencies(conn: sqlite3.Connection) -> None:
    """Unpivot cell_counts into the cell_frequencies summary table."""
    sql = """
        INSERT OR IGNORE INTO cell_frequencies
               (sample_id, population, count, total_count, percentage)
        SELECT sample_id, 'b_cell', b_cell, total,
               ROUND(b_cell * 100.0 / total, 4)
        FROM (SELECT *, b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte AS total
              FROM cell_counts)
        UNION ALL
        SELECT sample_id, 'cd8_t_cell', cd8_t_cell, total,
               ROUND(cd8_t_cell * 100.0 / total, 4)
        FROM (SELECT *, b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte AS total
              FROM cell_counts)
        UNION ALL
        SELECT sample_id, 'cd4_t_cell', cd4_t_cell, total,
               ROUND(cd4_t_cell * 100.0 / total, 4)
        FROM (SELECT *, b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte AS total
              FROM cell_counts)
        UNION ALL
        SELECT sample_id, 'nk_cell', nk_cell, total,
               ROUND(nk_cell * 100.0 / total, 4)
        FROM (SELECT *, b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte AS total
              FROM cell_counts)
        UNION ALL
        SELECT sample_id, 'monocyte', monocyte, total,
               ROUND(monocyte * 100.0 / total, 4)
        FROM (SELECT *, b_cell+cd8_t_cell+cd4_t_cell+nk_cell+monocyte AS total
              FROM cell_counts)
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        log.info("  cell_frequencies: %d rows inserted", cur.rowcount)
    except sqlite3.DatabaseError as e:
        log.error("Failed to populate cell_frequencies: %s", e)
        raise


def main() -> None:
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            log.info("Removed existing %s", DB_PATH)
    except OSError as e:
        log.error("Could not remove existing database '%s': %s", DB_PATH, e)
        raise

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
    except sqlite3.OperationalError as e:
        log.error("Failed to connect to database '%s': %s", DB_PATH, e)
        raise

    try:
        log.info("Initialising schema …")
        init_db(conn)

        log.info("Loading data from %s …", CSV_PATH)
        load_csv(conn, CSV_PATH)

        log.info("Populating cell_frequencies …")
        populate_frequencies(conn)

        # Quick sanity check
        try:
            cur = conn.cursor()
            for table in ("projects", "subjects", "samples", "cell_counts", "cell_frequencies"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                log.info("  %s: %d total rows", table, cur.fetchone()[0])
        except sqlite3.DatabaseError as e:
            log.error("Sanity check query failed: %s", e)
            raise

        log.info("Database ready: %s", DB_PATH)

    except Exception as e:
        log.error("Pipeline failed: %s", e)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
