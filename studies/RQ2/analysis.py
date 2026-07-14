import os
import sys
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_study_dir(config):
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ2")


def run_rq2_ols(df, y_var, x_var, controls):
    features = [x_var] + [c for c in controls if c in df.columns]
    reg_df = df[[y_var] + features].dropna().copy()
    if len(reg_df) < 10:
        return {"converged": False, "error": "Insufficient data"}
    X = sm.add_constant(reg_df[features])
    y = reg_df[y_var]
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
        return {
            "converged": True,
            "x_var": x_var,
            "coef": model.params.get(x_var, None),
            "std_err": model.bse.get(x_var, None),
            "pvalue": model.pvalues.get(x_var, None),
            "ci_lower": model.conf_int().loc[x_var, 0] if x_var in model.conf_int().index else None,
            "ci_upper": model.conf_int().loc[x_var, 1] if x_var in model.conf_int().index else None,
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        return {"converged": False, "error": str(e)}


def save_results_csv(results, filepath):
    rows = []
    for label, r in results.items():
        row = {"label": label}
        for k, v in r.items():
            if k == "model":
                continue
            row[k] = v
        rows.append(row)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)
    logger.info(f"Results saved: {filepath}")


def plot_forest_chart(results, title="RQ2: Knowledge & Data Atypicality",
                      output_path=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    fig, ax = plt.subplots(figsize=(10, 7))

    y_labels = []
    y_positions = []
    valid_items = [(label, r) for label, r in results.items() if r.get("converged", False)]

    for i, (label, res) in enumerate(reversed(valid_items)):
        coef = res["coef"]
        ci_lower = res["ci_lower"]
        ci_upper = res["ci_upper"]
        pval = res.get("pvalue", 1.0)

        y_pos = i
        y_positions.append(y_pos)
        y_labels.append(label)

        color = "#2166ac" if pval < 0.05 else "#cccccc"
        alpha = 1.0 if pval < 0.05 else 0.5

        ax.errorbar(coef, y_pos, xerr=[[coef - ci_lower], [ci_upper - coef]],
                     fmt="o", color=color, alpha=alpha, capsize=4, markersize=8,
                     ecolor="#333333", elinewidth=1.5)

        sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        text = f"{coef:.4f}{sig_str}" if pval < 0.05 else f"{coef:.4f} (n.s.)"
        x_text = ci_upper + 0.005 if coef >= 0 else ci_lower - 0.005
        ha = "left" if coef >= 0 else "right"
        ax.text(x_text, y_pos, text, va="center", ha=ha, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("Coefficient (95% CI)", fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sig_patch = mpatches.Patch(facecolor="#2166ac", alpha=0.85, edgecolor="black", label="Significant (p ≤ 0.05)")
    ns_patch = mpatches.Patch(facecolor="#cccccc", alpha=0.5, edgecolor="black", label="Non-significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", fontsize=11, frameon=False)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved: {output_path}")
    plt.close(fig)
    return fig


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    study_dir = get_study_dir(config)

    all_author_path = os.path.join(study_dir, "rq2_all_author_ready.csv")
    first_author_path = os.path.join(study_dir, "rq2_first_author_ready.csv")

    if not os.path.exists(all_author_path):
        logger.error(f"Data file not found: {all_author_path}")
        sys.exit(1)

    df_all = pd.read_csv(all_author_path)
    logger.info(f"Loaded RQ2 all-author data: {df_all.shape}")

    df_first = None
    if os.path.exists(first_author_path):
        df_first = pd.read_csv(first_author_path)
        logger.info(f"Loaded RQ2 first-author data: {df_first.shape}")

    x_vars = ["topic_distance", "dataset_distance"]
    controls = ["H_index", "Academic_Age", "Average_Team_Size", "Topic_Diversity"]
    y_vars = ["C3", "C5"]

    all_author_results = {}
    for y_var in y_vars:
        for x_var in x_vars:
            if y_var in df_all.columns and x_var in df_all.columns:
                label = f"{y_var} ~ {x_var}"
                all_author_results[label] = run_rq2_ols(df_all, y_var, x_var, controls)
                r = all_author_results[label]
                if r.get("converged"):
                    logger.info(f"All-author {label}: coef={r['coef']:.4f}, p={r['pvalue']:.6f}")

    results_dir = os.path.join(study_dir, "results")
    save_results_csv(all_author_results, os.path.join(results_dir, "rq2_all_author_results.csv"))

    first_author_results = {}
    if df_first is not None:
        for y_var in y_vars:
            for x_var in x_vars:
                if y_var in df_first.columns and x_var in df_first.columns:
                    label = f"{y_var} ~ {x_var}"
                    first_author_results[label] = run_rq2_ols(df_first, y_var, x_var, controls)
                    r = first_author_results[label]
                    if r.get("converged"):
                        logger.info(f"First-author {label}: coef={r['coef']:.4f}, p={r['pvalue']:.6f}")

        save_results_csv(first_author_results, os.path.join(results_dir, "rq2_first_author_results.csv"))

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_forest_chart(
        all_author_results,
        title="RQ2: Knowledge & Data Atypicality (All Authors)",
        output_path=os.path.join(figures_dir, "rq2_forest_all.pdf"),
    )

    if first_author_results:
        plot_forest_chart(
            first_author_results,
            title="RQ2: Knowledge & Data Atypicality (First Author)",
            output_path=os.path.join(figures_dir, "rq2_forest_first_author.pdf"),
        )

    logger.info("RQ2 analysis complete.")


if __name__ == "__main__":
    main()