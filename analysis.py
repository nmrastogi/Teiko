"""
analysis.py – Parts 2-4 of the Teiko clinical-trial analysis.

Outputs written to ./outputs/:
  part2_summary_table.csv   – relative frequency of each cell population per sample
  part3_boxplot.png         – responder vs non-responder boxplots (melanoma/miraclib/PBMC)
  part3_stats.csv           – Mann-Whitney U p-values + significance flags
  part4_results.txt         – baseline subset counts and average B-cell answer
"""

import json
import os
import sqlite3
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DB_PATH = "teiko.db"
OUT_DIR = "outputs"

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_output_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 2 – summary table of relative frequencies
# ---------------------------------------------------------------------------

def build_summary_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return long-form summary: one row per (sample, population)."""
    query = """
        SELECT
            s.sample_id                    AS sample,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM samples s
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        ORDER BY s.sample_id
    """
    wide = pd.read_sql_query(query, conn)

    wide["total_count"] = wide[POPULATIONS].sum(axis=1)

    rows = []
    for _, row in wide.iterrows():
        for pop in POPULATIONS:
            rows.append(
                {
                    "sample": row["sample"],
                    "total_count": int(row["total_count"]),
                    "population": pop,
                    "count": int(row[pop]),
                    "percentage": round(row[pop] / row["total_count"] * 100, 4),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3 – statistical comparison: responders vs non-responders
# ---------------------------------------------------------------------------

def load_melanoma_miraclib_pbmc(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return percentage data for melanoma / miraclib / PBMC samples only."""
    query = """
        SELECT
            s.sample_id,
            s.response,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        WHERE sub.condition    = 'melanoma'
          AND s.treatment      = 'miraclib'
          AND s.sample_type    = 'PBMC'
          AND s.response       IS NOT NULL
          AND s.response       != ''
        ORDER BY s.sample_id
    """
    df = pd.read_sql_query(query, conn)
    df["total_count"] = df[POPULATIONS].sum(axis=1)
    for pop in POPULATIONS:
        df[f"{pop}_pct"] = df[pop] / df["total_count"] * 100
    return df


def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U test per population; Benjamini-Hochberg FDR correction."""
    responders     = df[df["response"] == "yes"]
    non_responders = df[df["response"] == "no"]

    records = []
    for pop in POPULATIONS:
        col = f"{pop}_pct"
        stat, pval = stats.mannwhitneyu(
            responders[col].values,
            non_responders[col].values,
            alternative="two-sided",
        )
        records.append(
            {
                "population": pop,
                "n_responders": len(responders),
                "n_non_responders": len(non_responders),
                "median_responders": round(responders[col].median(), 4),
                "median_non_responders": round(non_responders[col].median(), 4),
                "mannwhitney_U": round(stat, 2),
                "p_value": round(pval, 6),
            }
        )

    stats_df = pd.DataFrame(records)

    # FDR correction (Benjamini-Hochberg)
    reject, pvals_corrected, _, _ = multipletests(
        stats_df["p_value"].values, alpha=0.05, method="fdr_bh"
    )
    stats_df["p_value_fdr"] = [round(p, 6) for p in pvals_corrected]
    stats_df["significant_fdr"] = reject

    return stats_df


def plot_boxplots(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: str) -> None:
    """Boxplot of relative frequencies per population, coloured by response."""
    fig, axes = plt.subplots(1, len(POPULATIONS), figsize=(16, 6), sharey=False)
    fig.suptitle(
        "Cell Population Relative Frequencies\nMelanoma · Miraclib · PBMC  –  Responders vs Non-Responders",
        fontsize=13,
        fontweight="bold",
    )

    colors = {"yes": "#2196F3", "no": "#F44336"}
    labels = {"yes": "Responder", "no": "Non-Responder"}

    for ax, pop in zip(axes, POPULATIONS):
        col = f"{pop}_pct"
        data_yes = df.loc[df["response"] == "yes", col].values
        data_no  = df.loc[df["response"] == "no",  col].values

        bp = ax.boxplot(
            [data_yes, data_no],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, resp in zip(bp["boxes"], ["yes", "no"]):
            patch.set_facecolor(colors[resp])
            patch.set_alpha(0.75)

        # Significance annotation
        row = stats_df[stats_df["population"] == pop].iloc[0]
        p_fdr = row["p_value_fdr"]
        if p_fdr < 0.001:
            sig_label = "***"
        elif p_fdr < 0.01:
            sig_label = "**"
        elif p_fdr < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        y_max = max(data_yes.max(), data_no.max()) * 1.05
        ax.annotate(
            f"p={row['p_value']:.4f}\n(FDR {p_fdr:.4f}) {sig_label}",
            xy=(1.5, y_max),
            ha="center",
            fontsize=7.5,
            color="black",
        )

        pop_label = pop.replace("_", " ").title()
        ax.set_title(pop_label, fontsize=10, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Responder", "Non-Resp."], fontsize=8)
        ax.set_ylabel("Relative frequency (%)" if pop == POPULATIONS[0] else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(facecolor=colors["yes"], alpha=0.75, label="Responder"),
        mpatches.Patch(facecolor=colors["no"],  alpha=0.75, label="Non-Responder"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Boxplot saved: {out_path}")


# ---------------------------------------------------------------------------
# Part 4 – baseline subset analysis
# ---------------------------------------------------------------------------

def run_part4(conn: sqlite3.Connection) -> str:
    """Query and summarise the melanoma / miraclib / PBMC / baseline subset."""

    # Base subset: melanoma PBMC at time=0 treated with miraclib
    base_query = """
        SELECT
            s.sample_id,
            sub.subject_id,
            sub.project_id,
            sub.condition,
            sub.sex,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start,
            cc.b_cell
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        WHERE sub.condition               = 'melanoma'
          AND s.sample_type               = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND s.treatment                 = 'miraclib'
        ORDER BY s.sample_id
    """
    df = pd.read_sql_query(base_query, conn)

    total_samples   = len(df)
    total_subjects  = df["subject_id"].nunique()

    # Samples per project
    samples_per_project = df.groupby("project_id").size().to_dict()

    # Subjects per response category (each subject appears once at baseline)
    response_counts = df.groupby("response")["subject_id"].nunique().to_dict()

    # Subjects per sex
    sex_counts = df.groupby("sex")["subject_id"].nunique().to_dict()

    # Average B cells: melanoma males, responders, time=0
    male_resp = df[(df["sex"] == "M") & (df["response"] == "yes")]
    avg_b_cell_male_responders = male_resp["b_cell"].mean()

    lines = [
        "=" * 60,
        "PART 4 – BASELINE SUBSET ANALYSIS",
        "Filter: melanoma · PBMC · time_from_treatment_start = 0 · miraclib",
        "=" * 60,
        "",
        f"Total samples in subset : {total_samples}",
        f"Total unique subjects   : {total_subjects}",
        "",
        "Samples per project:",
        *[f"  {proj}: {n}" for proj, n in sorted(samples_per_project.items())],
        "",
        "Subjects by response status:",
        *[f"  {resp}: {n}" for resp, n in sorted(response_counts.items())],
        "",
        "Subjects by sex:",
        *[f"  {sex}: {n}" for sex, n in sorted(sex_counts.items())],
        "",
        "-" * 60,
        "Average B-cell count for melanoma MALE RESPONDERS at time=0:",
        f"  {avg_b_cell_male_responders:.2f}",
        "  (n={})".format(len(male_resp)),
        "-" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export for D3.js dashboard
# ---------------------------------------------------------------------------

POP_LABELS = {
    "b_cell":      "B Cell",
    "cd8_t_cell":  "CD8 T Cell",
    "cd4_t_cell":  "CD4 T Cell",
    "nk_cell":     "NK Cell",
    "monocyte":    "Monocyte",
}


def export_json(conn: sqlite3.Connection) -> None:
    """Export three JSON files consumed by the D3.js dashboard."""

    # ---- data_part2.json ----
    query = """
        SELECT
            s.sample_id AS sample,
            sub.project_id  AS project,
            sub.condition,
            sub.sex,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start AS time,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        ORDER BY s.sample_id
    """
    wide = pd.read_sql_query(query, conn)
    wide["total_count"] = wide[POPULATIONS].sum(axis=1)

    samples_json = []
    for _, row in wide.iterrows():
        pops = {}
        for pop in POPULATIONS:
            pops[pop] = {
                "count": int(row[pop]),
                "percentage": round(row[pop] / row["total_count"] * 100, 4),
            }
        samples_json.append({
            "sample":       row["sample"],
            "project":      row["project"],
            "condition":    row["condition"],
            "sex":          row["sex"],
            "treatment":    row["treatment"],
            "response":     row["response"] if row["response"] else None,
            "sample_type":  row["sample_type"],
            "time":         int(row["time"]),
            "total_count":  int(row["total_count"]),
            "populations":  pops,
        })

    out2 = os.path.join(OUT_DIR, "data_part2.json")
    with open(out2, "w") as fh:
        json.dump({"samples": samples_json}, fh)
    print(f"  Part 2 JSON saved: {out2}")

    # ---- data_part3.json ----
    melanoma_df = load_melanoma_miraclib_pbmc(conn)
    stats_df    = run_statistics(melanoma_df)

    stats_list = []
    for _, row in stats_df.iterrows():
        stats_list.append({
            "population":       row["population"],
            "label":            POP_LABELS[row["population"]],
            "n_responders":     int(row["n_responders"]),
            "n_non_responders": int(row["n_non_responders"]),
            "median_resp":      float(row["median_responders"]),
            "median_nonresp":   float(row["median_non_responders"]),
            "p_value":          float(row["p_value"]),
            "p_value_fdr":      float(row["p_value_fdr"]),
            "significant_fdr":  bool(row["significant_fdr"]),
        })

    boxplot_data = {}
    for pop in POPULATIONS:
        col = f"{pop}_pct"
        boxplot_data[pop] = {
            "yes": [round(v, 4) for v in melanoma_df.loc[melanoma_df.response == "yes", col].tolist()],
            "no":  [round(v, 4) for v in melanoma_df.loc[melanoma_df.response == "no",  col].tolist()],
        }

    out3 = os.path.join(OUT_DIR, "data_part3.json")
    with open(out3, "w") as fh:
        json.dump({"stats": stats_list, "boxplot": boxplot_data}, fh)
    print(f"  Part 3 JSON saved: {out3}")

    # ---- data_part4.json ----
    part4_query = """
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
        WHERE sub.condition               = 'melanoma'
          AND s.sample_type               = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND s.treatment                 = 'miraclib'
    """
    p4 = pd.read_sql_query(part4_query, conn)
    male_resp = p4[(p4["sex"] == "M") & (p4["response"] == "yes")]

    p4_json = {
        "total_samples":           int(len(p4)),
        "total_subjects":          int(p4["subject_id"].nunique()),
        "samples_per_project":     {k: int(v) for k, v in p4.groupby("project_id").size().to_dict().items()},
        "subjects_by_response":    {k: int(v) for k, v in p4.groupby("response")["subject_id"].nunique().to_dict().items()},
        "subjects_by_sex":         {k: int(v) for k, v in p4.groupby("sex")["subject_id"].nunique().to_dict().items()},
        "avg_b_cell_male_resp":    round(float(male_resp["b_cell"].mean()), 2),
        "n_male_resp":             int(len(male_resp)),
    }

    out4 = os.path.join(OUT_DIR, "data_part4.json")
    with open(out4, "w") as fh:
        json.dump(p4_json, fh, indent=2)
    print(f"  Part 4 JSON saved: {out4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir()
    conn = get_connection()

    # ------ Part 2 ------
    print("Part 2: building summary table …")
    summary = build_summary_table(conn)
    out2 = os.path.join(OUT_DIR, "part2_summary_table.csv")
    summary.to_csv(out2, index=False)
    print(f"  Summary table saved: {out2}  ({len(summary)} rows)")

    # ------ Part 3 ------
    print("\nPart 3: statistical analysis …")
    melanoma_df = load_melanoma_miraclib_pbmc(conn)
    print(
        f"  Samples: {len(melanoma_df)}  "
        f"(responders={len(melanoma_df[melanoma_df.response=='yes'])}, "
        f"non-responders={len(melanoma_df[melanoma_df.response=='no'])})"
    )

    stats_df = run_statistics(melanoma_df)
    out3_stats = os.path.join(OUT_DIR, "part3_stats.csv")
    stats_df.to_csv(out3_stats, index=False)
    print(f"  Stats table saved: {out3_stats}")

    print("\n  Statistical results (Mann-Whitney U + BH-FDR correction):")
    print(f"  {'Population':<15} {'p-value':>10} {'p-FDR':>10} {'Significant':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    for _, row in stats_df.iterrows():
        sig = "YES *" if row["significant_fdr"] else "no"
        print(
            f"  {row['population']:<15} {row['p_value']:>10.6f} "
            f"{row['p_value_fdr']:>10.6f} {sig:>12}"
        )

    out3_plot = os.path.join(OUT_DIR, "part3_boxplot.png")
    plot_boxplots(melanoma_df, stats_df, out3_plot)

    # ------ Part 4 ------
    print("\nPart 4: baseline subset analysis …")
    report = run_part4(conn)
    print(report)
    out4 = os.path.join(OUT_DIR, "part4_results.txt")
    with open(out4, "w") as fh:
        fh.write(report + "\n")
    print(f"\n  Part 4 report saved: {out4}")

    # ------ JSON export for D3 dashboard ------
    print("\nExporting JSON for D3 dashboard …")
    export_json(conn)

    conn.close()
    print("\nAnalysis complete. Outputs are in ./outputs/")


if __name__ == "__main__":
    main()
