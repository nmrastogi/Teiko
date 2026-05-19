"""
server.py – Flask server for the Teiko D3.js dashboard.

Routes:
  GET /                              → serves index.html
  GET /api/initial-analysis          → sample frequencies, filtered server-side
                                       query params: condition, treatment, sample_type, time, response, sex, population
  GET /api/initial-analysis/opts     → distinct values for Initial Analysis filter dropdowns
  GET /api/statistical-analysis      → responder vs non-responder stats + boxplot data
                                       query params: condition, treatment, sample_type
  GET /api/data-subset-analysis      → filtered baseline subset
                                       query params: condition, sample_type, time, treatment
  GET /api/data-subset-analysis/opts → distinct values for Data Subset Analysis filter dropdowns
"""

import logging
import sqlite3

import pandas as pd
from flask import Flask, jsonify, request, send_file
from scipy import stats

from analysis import get_connection, load_melanoma_miraclib_pbmc, run_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
POP_LABELS = {
    "b_cell":      "B Cell",
    "cd8_t_cell":  "CD8 T Cell",
    "cd4_t_cell":  "CD4 T Cell",
    "nk_cell":     "NK Cell",
    "monocyte":    "Monocyte",
}


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page D3.js dashboard."""
    return send_file("index.html")


# ---------------------------------------------------------------------------
# API – Initial Analysis filter options
# ---------------------------------------------------------------------------

@app.route("/api/initial-analysis/opts")
def api_initial_analysis_opts():
    """
    Return the distinct values for every Initial Analysis filter dropdown.

    Queried fresh from the database so the options always reflect what is
    actually present in teiko.db — no hard-coded lists.

    Response shape:
        {
            "conditions":   [...],   # e.g. ["carcinoma", "healthy", "melanoma"]
            "treatments":   [...],   # e.g. ["miraclib", "none", "phauximab"]
            "sample_types": [...],   # e.g. ["PBMC", "WB"]
            "times":        [...],   # e.g. [0, 7, 14]
            "responses":    [...],   # e.g. ["no", "yes"]
            "sexes":        [...]    # e.g. ["F", "M"]
        }
    """
    queries = {
        "conditions":   "SELECT DISTINCT condition FROM subjects ORDER BY condition",
        "treatments":   "SELECT DISTINCT treatment FROM samples ORDER BY treatment",
        "sample_types": "SELECT DISTINCT sample_type FROM samples ORDER BY sample_type",
        "times":        "SELECT DISTINCT time_from_treatment_start FROM samples ORDER BY time_from_treatment_start",
        "responses":    "SELECT DISTINCT response FROM samples WHERE response IS NOT NULL AND response != '' ORDER BY response",
        "sexes":        "SELECT DISTINCT sex FROM subjects ORDER BY sex",
    }
    try:
        conn = get_connection()
        try:
            result = {
                key: pd.read_sql_query(sql, conn).iloc[:, 0].tolist()
                for key, sql in queries.items()
            }
        except sqlite3.DatabaseError as e:
            log.error("Part 2 opts query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# API – Initial Analysis
# ---------------------------------------------------------------------------

@app.route("/api/initial-analysis")
def api_initial_analysis():
    """
    Return relative cell-population frequencies for the requested subset.

    All parameters are optional; omitting a parameter removes that filter so
    all values for that dimension are included.

    Query parameters:
        condition   – patient condition   (e.g. melanoma, carcinoma, healthy)
        treatment   – drug administered   (e.g. miraclib, phauximab, none)
        sample_type – sample material     (PBMC or WB)
        time        – days from treatment start (0, 7, or 14)
        response    – treatment response  (yes or no)
        sex         – patient sex         (M or F)
        population  – cell population     (b_cell, cd8_t_cell, …)

    Response shape:
        {
            "rows":  [ { sample, project, condition, sex, treatment, response,
                         sample_type, time, total_count,
                         population, count, percentage }, … ],
            "total": <int>   # total number of rows returned
        }
    """
    condition   = request.args.get("condition",   "")
    treatment   = request.args.get("treatment",   "")
    sample_type = request.args.get("sample_type", "")
    time        = request.args.get("time",        "")
    response    = request.args.get("response",    "")
    sex         = request.args.get("sex",         "")
    population  = request.args.get("population",  "")

    where_clauses, params = [], []
    if condition:   where_clauses.append("sub.condition = ?");                    params.append(condition)
    if treatment:   where_clauses.append("s.treatment = ?");                      params.append(treatment)
    if sample_type: where_clauses.append("s.sample_type = ?");                   params.append(sample_type)
    if time:        where_clauses.append("s.time_from_treatment_start = ?");      params.append(int(time))
    if response:    where_clauses.append("s.response = ?");                       params.append(response)
    if sex:         where_clauses.append("sub.sex = ?");                          params.append(sex)
    if population:  where_clauses.append("cf.population = ?");                    params.append(population)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT
            cf.sample_id   AS sample,
            sub.project_id AS project,
            sub.condition,
            sub.sex,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start AS time,
            cf.total_count,
            cf.population,
            cf.count,
            cf.percentage
        FROM cell_frequencies cf
        JOIN samples s    ON s.sample_id    = cf.sample_id
        JOIN subjects sub ON sub.subject_id = s.subject_id
        {where_sql}
        ORDER BY cf.sample_id, cf.population
    """
    try:
        conn = get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params if params else None)
        except sqlite3.DatabaseError as e:
            log.error("Part 2 query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

    rows = [
        {
            "sample":      r["sample"],
            "project":     r["project"],
            "condition":   r["condition"],
            "sex":         r["sex"],
            "treatment":   r["treatment"],
            "response":    None if pd.isna(r["response"]) else r["response"],
            "sample_type": r["sample_type"],
            "time":        int(r["time"]),
            "total_count": int(r["total_count"]),
            "population":  r["population"],
            "count":       int(r["count"]),
            "percentage":  float(r["percentage"]),
        }
        for _, r in df.iterrows()
    ]

    log.info("Part 2: %d rows returned (filters: %s)", len(rows),
             {k: v for k, v in [("condition", condition), ("treatment", treatment),
              ("sample_type", sample_type), ("time", time), ("response", response),
              ("sex", sex), ("population", population)] if v})
    return jsonify({"rows": rows, "total": len(rows)})


# ---------------------------------------------------------------------------
# API – Statistical Analysis
# ---------------------------------------------------------------------------

def _boxplot_from_sql(conn, condition, treatment, sample_type):
    """
    Compute boxplot statistics for all five cell populations entirely in SQL.

    Uses a four-step CTE chain so no data leaves the database engine until
    the final summary row per (population, response) group is ready:

      base      – ranks every percentage value within (population, response)
                  using ROW_NUMBER(); also counts rows per group.
      quartiles – picks the two rows bracketing the 25th / 50th / 75th
                  percentile index and averages them (linear interpolation).
      fences    – derives the Tukey fence bounds: Q1 − 1.5×IQR, Q3 + 1.5×IQR.
      whiskers  – clamps the fence to the nearest real data point on each side.
      final     – aggregates outlier values (points outside the whiskers) into
                  a comma-separated string via GROUP_CONCAT.

    Returns a nested dict keyed by population → response group ("yes"/"no"),
    each containing q1, median, q3, whisker_low, whisker_high, outliers.
    """
    query = """
    WITH base AS (
        -- Rank each percentage value within (population, response)
        SELECT
            cf.population,
            s.response,
            ROUND(cf.percentage, 4)                                                              AS pct,
            ROW_NUMBER() OVER (PARTITION BY cf.population, s.response ORDER BY cf.percentage)   AS rn,
            COUNT(*)     OVER (PARTITION BY cf.population, s.response)                          AS n
        FROM cell_frequencies cf
        JOIN samples s    ON s.sample_id    = cf.sample_id
        JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE sub.condition = ? AND s.treatment = ? AND s.sample_type = ?
          AND s.response IN ('yes','no')
    ),
    quartiles AS (
        -- Linear-interpolation quartiles using the two rows bracketing each percentile index
        SELECT population, response,
            AVG(CASE WHEN rn IN (MAX(1, CAST(n*0.25 AS INT)),
                                 MIN(n, CAST(n*0.25 AS INT)+1)) THEN pct END) AS q1,
            AVG(CASE WHEN rn IN (MAX(1, CAST(n*0.50 AS INT)),
                                 MIN(n, CAST(n*0.50 AS INT)+1)) THEN pct END) AS median,
            AVG(CASE WHEN rn IN (MAX(1, CAST(n*0.75 AS INT)),
                                 MIN(n, CAST(n*0.75 AS INT)+1)) THEN pct END) AS q3
        FROM base
        GROUP BY population, response
    ),
    fences AS (
        SELECT *,
            q1 - 1.5 * (q3 - q1) AS fence_low,
            q3 + 1.5 * (q3 - q1) AS fence_high
        FROM quartiles
    ),
    whiskers AS (
        -- Whisker endpoints are the most extreme real data points inside the fences
        SELECT f.population, f.response,
               ROUND(f.q1,4) AS q1, ROUND(f.median,4) AS median, ROUND(f.q3,4) AS q3,
               ROUND(MIN(CASE WHEN b.pct >= f.fence_low  THEN b.pct END), 4) AS whisker_low,
               ROUND(MAX(CASE WHEN b.pct <= f.fence_high THEN b.pct END), 4) AS whisker_high,
               f.fence_low, f.fence_high
        FROM fences f
        JOIN base b ON b.population = f.population AND b.response = f.response
        GROUP BY f.population, f.response
    )
    -- Final: summary row per (population, response) + outlier list via GROUP_CONCAT
    SELECT w.population, w.response,
           w.q1, w.median, w.q3, w.whisker_low, w.whisker_high,
           GROUP_CONCAT(
               CASE WHEN b.pct < w.whisker_low OR b.pct > w.whisker_high THEN b.pct END,
               ','
           ) AS outliers_csv
    FROM whiskers w
    JOIN base b ON b.population = w.population AND b.response = w.response
    GROUP BY w.population, w.response
    ORDER BY w.population, w.response
    """
    df = pd.read_sql_query(query, conn, params=(condition, treatment, sample_type))

    result = {}
    for _, row in df.iterrows():
        pop, resp = row["population"], row["response"]
        outliers = (
            [float(v) for v in str(row["outliers_csv"]).split(",")]
            if row["outliers_csv"] else []
        )
        result.setdefault(pop, {})[resp] = {
            "q1":           round(float(row["q1"]),          4),
            "median":       round(float(row["median"]),       4),
            "q3":           round(float(row["q3"]),           4),
            "whisker_low":  round(float(row["whisker_low"]),  4),
            "whisker_high": round(float(row["whisker_high"]), 4),
            "outliers":     outliers,
        }
    return result


@app.route("/api/statistical-analysis")
def api_statistical_analysis():
    """
    Run Mann-Whitney U tests comparing responders vs non-responders for each
    cell population in the selected subset, and return pre-computed boxplot
    statistics for the dashboard.

    Query parameters (all optional, defaults shown):
        condition   – patient condition  (default: melanoma)
        treatment   – drug administered  (default: miraclib)
        sample_type – sample material    (default: PBMC)

    Returns 404 if no samples match, 422 if only one response group is present
    (a two-group comparison requires both responders and non-responders).

    Response shape:
        {
            "stats":   [ { population, label, n_responders, n_non_responders,
                           median_resp, median_nonresp, p_value, significant }, … ],
            "boxplot": { "<population>": { "yes": { q1, median, q3,
                                                    whisker_low, whisker_high,
                                                    outliers }, "no": {…} } },
            "filters": { condition, treatment, sample_type }
        }
    """
    condition   = request.args.get("condition",   "melanoma")
    treatment   = request.args.get("treatment",   "miraclib")
    sample_type = request.args.get("sample_type", "PBMC")

    try:
        conn = get_connection()
        try:
            melanoma_df = load_melanoma_miraclib_pbmc(conn, condition, treatment, sample_type)
            boxplot     = _boxplot_from_sql(conn, condition, treatment, sample_type)
        except sqlite3.DatabaseError as e:
            log.error("Part 3 query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

    if melanoma_df.empty:
        return jsonify({"error": "No samples match the selected filters."}), 404

    resp_groups = melanoma_df["response"].unique()
    if "yes" not in resp_groups or "no" not in resp_groups:
        return jsonify({"error": "Both responder and non-responder groups are required for statistical comparison."}), 422

    stats_df = run_statistics(melanoma_df)

    stats_list = [
        {
            "population":       row["population"],
            "label":            POP_LABELS[row["population"]],
            "n_responders":     int(row["n_responders"]),
            "n_non_responders": int(row["n_non_responders"]),
            "median_resp":      float(row["median_responders"]),
            "median_nonresp":   float(row["median_non_responders"]),
            "p_value":          float(row["p_value"]),
            "significant":      bool(row["significant"]),
        }
        for _, row in stats_df.iterrows()
    ]

    log.info("Part 3: %s / %s / %s → %d samples", condition, treatment, sample_type, len(melanoma_df))
    return jsonify({"stats": stats_list, "boxplot": boxplot, "filters": {"condition": condition, "treatment": treatment, "sample_type": sample_type}})


# ---------------------------------------------------------------------------
# API – Data Subset Analysis filter options
# ---------------------------------------------------------------------------

@app.route("/api/data-subset-analysis/opts")
def api_data_subset_analysis_opts():
    """
    Return the distinct values for every Data Subset Analysis filter dropdown.

    Response shape:
        {
            "conditions":   [...],
            "sample_types": [...],
            "times":        [...],
            "treatments":   [...]
        }
    """
    queries = {
        "conditions":   "SELECT DISTINCT condition FROM subjects ORDER BY condition",
        "sample_types": "SELECT DISTINCT sample_type FROM samples ORDER BY sample_type",
        "times":        "SELECT DISTINCT time_from_treatment_start FROM samples ORDER BY time_from_treatment_start",
        "treatments":   "SELECT DISTINCT treatment FROM samples ORDER BY treatment",
    }
    try:
        conn = get_connection()
        try:
            result = {
                key: pd.read_sql_query(sql, conn).iloc[:, 0].tolist()
                for key, sql in queries.items()
            }
        except sqlite3.DatabaseError as e:
            log.error("Part 4 opts query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# API – Data Subset Analysis
# ---------------------------------------------------------------------------

@app.route("/api/data-subset-analysis")
def api_data_subset_analysis():
    """
    Return aggregated statistics for a filtered patient subset.

    Designed to answer exploratory questions such as how many samples came
    from each project, how subjects split by response and sex, and what the
    average B-cell count is for male responders.

    Query parameters (all optional, defaults shown):
        condition   – patient condition       (default: melanoma)
        sample_type – sample material         (default: PBMC)
        time        – days from treatment start (default: 0)
        treatment   – drug administered       (default: miraclib)
                      pass an empty string to include all treatments.

    Returns an empty-result payload (zeros / empty dicts) when no samples
    match rather than raising an error, so the dashboard degrades gracefully.

    Response shape:
        {
            "total_samples":        <int>,
            "total_subjects":       <int>,
            "samples_per_project":  { "<project_id>": <int>, … },
            "subjects_by_response": { "yes": <int>, "no": <int> },
            "subjects_by_sex":      { "M": <int>, "F": <int> },
            "avg_b_cell_male_resp": <float|null>,  # null if no male responders
            "n_male_resp":          <int>,
            "filters":              { condition, sample_type, time, treatment }
        }
    """
    condition   = request.args.get("condition",   "")
    sample_type = request.args.get("sample_type", "")
    time        = request.args.get("time",        "")
    treatment   = request.args.get("treatment",   "")

    try:
        time_int = int(time) if time else None
    except ValueError:
        return jsonify({"error": f"Invalid time value: {time}"}), 400

    where_clauses, params = [], []
    if condition:   where_clauses.append("sub.condition = ?");                   params.append(condition)
    if sample_type: where_clauses.append("s.sample_type = ?");                   params.append(sample_type)
    if time_int is not None: where_clauses.append("s.time_from_treatment_start = ?"); params.append(time_int)
    if treatment:   where_clauses.append("s.treatment = ?");                     params.append(treatment)

    query = f"""
        SELECT
            s.sample_id,
            sub.subject_id,
            sub.project_id,
            sub.sex,
            s.response,
            cc.b_cell
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        {"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
    """
    try:
        conn = get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
        except sqlite3.DatabaseError as e:
            log.error("Part 4 query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

    if df.empty:
        return jsonify({
            "total_samples": 0, "total_subjects": 0,
            "samples_per_project": {}, "subjects_by_response": {},
            "subjects_by_sex": {}, "avg_b_cell_male_resp": None, "n_male_resp": 0,
        })

    male_resp = df[(df["sex"] == "M") & (df["response"] == "yes")]
    avg_b = round(float(male_resp["b_cell"].mean()), 2) if not male_resp.empty else None

    result = {
        "total_samples":        int(len(df)),
        "total_subjects":       int(df["subject_id"].nunique()),
        "samples_per_project":  {k: int(v) for k, v in df.groupby("project_id").size().items()},
        "subjects_by_response": {k: int(v) for k, v in df.groupby("response")["subject_id"].nunique().items()},
        "subjects_by_sex":      {k: int(v) for k, v in df.groupby("sex")["subject_id"].nunique().items()},
        "avg_b_cell_male_resp": avg_b,
        "n_male_resp":          int(len(male_resp)),
        "filters":              {"condition": condition or None, "sample_type": sample_type or None, "time": time_int, "treatment": treatment or None},
    }

    log.info("Part 4: %s / %s / t=%s / %s → %d samples", condition, sample_type, time, treatment or "all", result["total_samples"])
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting Teiko dashboard server on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
