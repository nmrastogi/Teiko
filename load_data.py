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
import os
import sqlite3

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

CREATE INDEX IF NOT EXISTS idx_subjects_condition  ON subjects(condition);
CREATE INDEX IF NOT EXISTS idx_subjects_project    ON subjects(project_id);
CREATE INDEX IF NOT EXISTS idx_samples_treatment   ON samples(treatment);
CREATE INDEX IF NOT EXISTS idx_samples_response    ON samples(response);
CREATE INDEX IF NOT EXISTS idx_samples_type        ON samples(sample_type);
CREATE INDEX IF NOT EXISTS idx_samples_time        ON samples(time_from_treatment_start);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def load_csv(conn: sqlite3.Connection, csv_path: str = CSV_PATH) -> None:
    cursor = conn.cursor()
    inserted = {"projects": 0, "subjects": 0, "samples": 0, "cell_counts": 0}

    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
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

    conn.commit()
    for table, n in inserted.items():
        print(f"  {table}: {n} rows inserted")


def main() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    print("Initialising schema …")
    init_db(conn)

    print(f"Loading data from {CSV_PATH} …")
    load_csv(conn, CSV_PATH)

    # Quick sanity check
    cur = conn.cursor()
    for table in ("projects", "subjects", "samples", "cell_counts"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} total rows")

    conn.close()
    print(f"\nDatabase ready: {DB_PATH}")


if __name__ == "__main__":
    main()
