import os
import sys
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
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
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "Q1")


def run_q1_regressions(df):
    x_var = "Atypicality_of_datasets_original_1"
    controls = ["Average_Team_Size", "Avg_Citation_Without_Self", "Academic_Age", "Topic_Diversity"]
    controls = [c for c in controls if c in df.columns]

    nb_results = {}
    for y_var in ["H_index", "Productivity", "C3", "C5"]:
        if y_var not in df.columns:
            continue
        features = [x_var] + controls
        reg_df = df[[y_var] + features].dropna().copy()
        X = sm.add_constant(reg_df[features])
        y = reg_df[y_var]
        try:
            model = NegativeBinomial(y, X, log_like_method="nb2").fit(disp=0, maxiter=300)
            coef = model.params.get(x_var, None)
            se = model.bse.get(x_var, None)
            pval = model.pvalues.get(x_var, None)
            ci = model.conf_int()
            nb_results[y_var] = {
                "converged": True,
                "x_var": x_var,
                "coef": coef,
                "std_err": se,
                "pvalue": pval,
                "ci_lower": ci.loc[x_var, 0] if x_var in ci.index else None,
                "ci_upper": ci.loc[x_var, 1] if x_var in ci.index else None,
                "r_squared": getattr(model, "pseudo_rsquared", None),
                "n_obs": int(model.nobs),
            }
            if nb_results[y_var]["converged"]:
                logger.info(f"NB {y_var}: coef={coef:.4f}, p={pval:.6f}")
        except Exception as e:
            logger.warning(f"NB regression for {y_var} failed: {e}")
            nb_results[y_var] = {"converged": False, "error": str(e)}

    ols_results = {}
    for y_var in ["C3", "C5"]:
        if y_var not in df.columns:
            continue
        features = [x_var] + controls
        reg_df = df[[y_var] + features].dropna().copy()
        X = sm.add_constant(reg_df[features])
        y = reg_df[y_var]
        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            coef = model.params.get(x_var, None)
            se = model.bse.get(x_var, None)
            pval = model.pvalues.get(x_var, None)
            ci = model.conf_int()
            ols_results[y_var + "_OLS"] = {
                "converged": True,
                "x_var": x_var,
                "coef": coef,
                "std_err": se,
                "pvalue": pval,
                "ci_lower": ci.loc[x_var, 0] if x_var in ci.index else None,
                "ci_upper": ci.loc[x_var, 1] if x_var in ci.index else None,
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
            }
            logger.info(f"OLS {y_var}: coef={coef:.4f}, p={pval:.6f}")
        except Exception as e:
            logger.warning(f"OLS regression for {y_var} failed: {e}")
            ols_results[y_var + "_OLS"] = {"converged": False, "error": str(e)}

    all_results = {**nb_results, **ols_results}
    return all_results


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


def plot_forest_chart(results, title="Q1: Effect of Data Atypicality on Academic Success",
                      output_path=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    fig, ax = plt.subplots(figsize=(10, 6))

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

        display_label = label.replace("_OLS", " (OLS)").replace("_", " ")
        y_labels.append(display_label)

        color = "#2166ac" if pval < 0.05 else "#cccccc"
        alpha = 1.0 if pval < 0.05 else 0.5

        ax.errorbar(coef, y_pos, xerr=[[coef - ci_lower], [ci_upper - coef]],
                     fmt="o", color=color, alpha=alpha, capsize=4, markersize=8,
                     ecolor="#333333", elinewidth=1.5)

        sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        text = f"{coef:.4f}{sig_str}" if pval < 0.05 else f"{coef:.4f} (n.s.)"
        x_text = ci_upper + 0.005 if coef >= 0 else ci_lower - 0.005
        ha = "left" if coef >= 0 else "right"
        ax.text(x_text, y_pos, text, va="center", ha=ha, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=13)
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

    data_path = os.path.join(study_dir, "q1_regression_ready.csv")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded Q1 data: {df.shape}")

    results = run_q1_regressions(df)

    results_path = os.path.join(study_dir, "results", "q1_regression_results.csv")
    save_results_csv(results, results_path)

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_forest_chart(
        results,
        title="Q1: Effect of Data Atypicality on Academic Success",
        output_path=os.path.join(figures_dir, "q1_forest_plot.pdf"),
    )

    logger.info("Q1 analysis complete.")


if __name__ == "__main__":
    main()