"""
analysis.py – Parts 2-4 of the Teiko clinical-trial analysis.

Outputs written to ./outputs/:
  part2_summary_table.csv   – relative frequency of each cell population per sample
  part3_boxplot.png         – responder vs non-responder boxplots (melanoma/miraclib/PBMC)
  part3_stats.csv           – Mann-Whitney U p-values + significance flags
  part4_results.txt         – baseline subset counts and average B-cell answer
"""

import logging
import os
import sqlite3
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats

DB_PATH = "teiko.db"
OUT_DIR = "outputs"

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.OperationalError as e:
        log.error("Failed to connect to database '%s': %s", DB_PATH, e)
        raise


def ensure_output_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 2 – summary table of relative frequencies
# ---------------------------------------------------------------------------

def build_summary_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return long-form summary: one row per (sample, population)."""
    query = """
        SELECT
            cf.sample_id  AS sample,
            cf.total_count,
            cf.population,
            cf.count,
            cf.percentage
        FROM cell_frequencies cf
        ORDER BY cf.sample_id, cf.population
    """
    try:
        return pd.read_sql_query(query, conn)
    except sqlite3.DatabaseError as e:
        log.error("Query failed in build_summary_table: %s", e)
        raise


# ---------------------------------------------------------------------------
# Part 3 – statistical comparison: responders vs non-responders
# ---------------------------------------------------------------------------

def load_melanoma_miraclib_pbmc(
    conn: sqlite3.Connection,
    condition: str = "melanoma",
    treatment: str = "miraclib",
    sample_type: str = "PBMC",
) -> pd.DataFrame:
    """Return percentage data for the requested condition / treatment / sample_type subset."""
    query = """
        SELECT
            s.sample_id,
            s.response,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        WHERE sub.condition    = ?
          AND s.treatment      = ?
          AND s.sample_type    = ?
          AND s.response       IS NOT NULL
          AND s.response       != ''
        ORDER BY s.sample_id
    """
    try:
        df = pd.read_sql_query(query, conn, params=(condition, treatment, sample_type))
    except sqlite3.DatabaseError as e:
        log.error("Query failed in load_melanoma_miraclib_pbmc: %s", e)
        raise
    df["total_count"] = df[POPULATIONS].sum(axis=1)
    for pop in POPULATIONS:
        df[f"{pop}_pct"] = df[pop] / df["total_count"] * 100
    return df


def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U test per population; significance at α = 0.05 on raw p-values."""
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
                "significant": bool(pval < 0.05),
            }
        )

    return pd.DataFrame(records)


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
        pval = row["p_value"]
        if pval < 0.001:
            sig_label = "***"
        elif pval < 0.01:
            sig_label = "**"
        elif pval < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        y_max = max(data_yes.max(), data_no.max()) * 1.05
        ax.annotate(
            f"p={pval:.4f} {sig_label}",
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
    log.info("Boxplot saved: %s", out_path)


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
    try:
        df = pd.read_sql_query(base_query, conn)
    except sqlite3.DatabaseError as e:
        log.error("Query failed in run_part4: %s", e)
        raise

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir()
    conn = get_connection()
    try:
        # ------ Part 2 ------
        log.info("Part 2: building summary table …")
        summary = build_summary_table(conn)
        out2 = os.path.join(OUT_DIR, "part2_summary_table.csv")
        summary.to_csv(out2, index=False)
        log.info("Summary table saved: %s  (%d rows)", out2, len(summary))

        # ------ Part 3 ------
        log.info("Part 3: statistical analysis …")
        melanoma_df = load_melanoma_miraclib_pbmc(conn)
        log.info(
            "Samples: %d  (responders=%d, non-responders=%d)",
            len(melanoma_df),
            len(melanoma_df[melanoma_df.response == "yes"]),
            len(melanoma_df[melanoma_df.response == "no"]),
        )

        stats_df = run_statistics(melanoma_df)
        out3_stats = os.path.join(OUT_DIR, "part3_stats.csv")
        stats_df.to_csv(out3_stats, index=False)
        log.info("Stats table saved: %s", out3_stats)

        log.info("Statistical results (Mann-Whitney U, α = 0.05):")
        log.info("  %-15s %10s %12s", "Population", "p-value", "Significant")
        log.info("  %s %s %s", "-"*15, "-"*10, "-"*12)
        for _, row in stats_df.iterrows():
            sig = "YES *" if row["significant"] else "no"
            log.info(
                "  %-15s %10.6f %12s",
                row["population"], row["p_value"], sig,
            )

        out3_plot = os.path.join(OUT_DIR, "part3_boxplot.png")
        plot_boxplots(melanoma_df, stats_df, out3_plot)

        # ------ Part 4 ------
        log.info("Part 4: baseline subset analysis …")
        report = run_part4(conn)
        log.info("%s", report)
        out4 = os.path.join(OUT_DIR, "part4_results.txt")
        with open(out4, "w") as fh:
            fh.write(report + "\n")
        log.info("Part 4 report saved: %s", out4)

        log.info("Analysis complete. Outputs are in ./outputs/")

    except Exception as e:
        log.error("Analysis failed: %s", e)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
