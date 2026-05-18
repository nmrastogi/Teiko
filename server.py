"""
server.py – Flask server for the Teiko D3.js dashboard.

Routes:
  GET /                → serves index.html
  GET /api/part2       → sample frequencies, filtered server-side
                         query params: condition, treatment, sample_type, time, response, sex, population
  GET /api/part2/opts  → distinct values for Part 2 filter dropdowns
  GET /api/part3       → responder vs non-responder stats + boxplot data
  GET /api/part4       → filtered baseline subset
                         query params: condition, sample_type, time, treatment
  GET /api/part4/opts  → distinct values for Part 4 filter dropdowns
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
    return send_file("index.html")


# ---------------------------------------------------------------------------
# API – Part 2 filter options
# ---------------------------------------------------------------------------

@app.route("/api/part2/opts")
def api_part2_opts():
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
# API – Part 2
# ---------------------------------------------------------------------------

@app.route("/api/part2")
def api_part2():
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
# API – Part 3
# ---------------------------------------------------------------------------

@app.route("/api/part3")
def api_part3():
    try:
        conn = get_connection()
        try:
            melanoma_df = load_melanoma_miraclib_pbmc(conn)
        except sqlite3.DatabaseError as e:
            log.error("Part 3 query failed: %s", e)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return jsonify({"error": str(e)}), 500

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

    boxplot = {
        pop: {
            "yes": [round(v, 4) for v in melanoma_df.loc[melanoma_df.response == "yes", f"{pop}_pct"].tolist()],
            "no":  [round(v, 4) for v in melanoma_df.loc[melanoma_df.response == "no",  f"{pop}_pct"].tolist()],
        }
        for pop in POPULATIONS
    }

    log.info("Part 3: returned stats for %d populations", len(stats_list))
    return jsonify({"stats": stats_list, "boxplot": boxplot})


# ---------------------------------------------------------------------------
# API – Part 4 filter options
# ---------------------------------------------------------------------------

@app.route("/api/part4/opts")
def api_part4_opts():
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
# API – Part 4
# ---------------------------------------------------------------------------

@app.route("/api/part4")
def api_part4():
    condition   = request.args.get("condition",   "melanoma")
    sample_type = request.args.get("sample_type", "PBMC")
    time        = request.args.get("time",        "0")
    treatment   = request.args.get("treatment",   "miraclib")

    try:
        time_int = int(time)
    except ValueError:
        return jsonify({"error": f"Invalid time value: {time}"}), 400

    where_clauses = ["sub.condition = ?", "s.sample_type = ?", "s.time_from_treatment_start = ?"]
    params = [condition, sample_type, time_int]
    if treatment:
        where_clauses.append("s.treatment = ?")
        params.append(treatment)

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
        WHERE {" AND ".join(where_clauses)}
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
        "filters":              {"condition": condition, "sample_type": sample_type, "time": time_int, "treatment": treatment},
    }

    log.info("Part 4: %s / %s / t=%s / %s → %d samples", condition, sample_type, time, treatment or "all", result["total_samples"])
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting Teiko dashboard server on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
