import os
import sys
import logging
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_study_dir(config):
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ4")


def winsorize_series(s, limits=(0.01, 0.99)):
    return s.clip(lower=s.quantile(limits[0]), upper=s.quantile(limits[1]))


def prepare_rq4_data(config):
    q2_path = config["paths"]["interim_q2"]
    csv_path = os.path.join(PROJECT_ROOT, q2_path, "atypicality_authorlevel_withCountry.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        sys.exit(1)

    df_raw = pd.read_csv(csv_path)
    logger.info(f"Loaded atypicality_authorlevel_withCountry: {df_raw.shape}")

    df = df_raw.dropna(subset=["Hemisphere", "Develop_Status"]).copy()
    df = df[(df["Hemisphere"] != "Unknown") & (df["Develop_Status"] != "Unknown")]
    logger.info(f"After filtering Unknown: {df.shape}")

    to_win = [
        "H_index", "Total_Papers", "Average_Institution_Citation",
        "average_data_cite", "author_avg_journal_impact",
        "Academic_Age", "Average_Team_Size", "unique_data_count",
    ]
    for col in to_win:
        if col in df.columns:
            df[f"{col}_w"] = winsorize_series(df[col])

    log_cols = [
        "H_index_w", "Total_Papers_w", "Average_Institution_Citation_w",
        "average_data_cite_w", "author_avg_journal_impact_w",
        "Academic_Age_w", "Average_Team_Size_w", "unique_data_count_w",
        "avg_time_period",
    ]
    for col in log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    for col in ["Atypicality_of_datasets_original_1"]:
        if col in df.columns:
            df[f"{col}_c"] = df[col] - df[col].mean()

    return df


def run_rq4_regressions(df):
    variables = [
        "log_H_index_w",
        "log_Total_Papers_w",
        "log_Average_Institution_Citation_w",
        "Avg_Citation_Without_Self",
        "Topic_Diversity",
        "log_average_data_cite_w",
        "log_author_avg_journal_impact_w",
        "log_avg_time_period",
        "log_Academic_Age_w",
        "log_Average_Team_Size_w",
        "log_unique_data_count_w",
    ]
    variables = [v for v in variables if v in df.columns]
    variable_str = " + ".join(variables)

    formula_hemi = f"Atypicality_of_datasets_original_1 ~ C(Hemisphere, Treatment(reference='South')) + {variable_str}"
    model_hemi = smf.ols(formula_hemi, data=df).fit(cov_type="HC3")
    logger.info(f"RQ4 Hemisphere model: R²={model_hemi.rsquared:.4f}, n={int(model_hemi.nobs)}")

    formula_dev = f"Atypicality_of_datasets_original_1 ~ C(Develop_Status, Treatment(reference='Developed')) + {variable_str}"
    model_dev = smf.ols(formula_dev, data=df).fit(cov_type="HC3")
    logger.info(f"RQ4 Develop_Status model: R²={model_dev.rsquared:.4f}, n={int(model_dev.nobs)}")

    return model_hemi, model_dev


def extract_rq4_results(model_hemi, model_dev, df):
    var_map = {
        "log_H_index_w": "H-index",
        "log_Total_Papers_w": "Total Papers",
        "log_Academic_Age_w": "Academic Age",
        "log_Average_Team_Size_w": "Team Size",
        "log_author_avg_journal_impact_w": "Average Journal Impact",
        "log_unique_data_count_w": "Unique Data Count",
        "Topic_Diversity": "Topic Diversity",
        "log_Average_Institution_Citation_w": "Average Institution Citation",
        "Avg_Citation_Without_Self": "Average Citation Without Self",
        "log_average_data_cite_w": "Average Data Citation",
        "log_avg_time_period": "Average Time Period",
        "C(Hemisphere, Treatment(reference='South'))[T.North]": "North (vs. South)",
        "C(Hemisphere, Treatment(reference='South'))[T.Equatorial]": "Equatorial (vs. South)",
        "C(Hemisphere, Treatment(reference='South'))[T.Mixed]": "Mixed Hemisphere (vs. South)",
        "C(Develop_Status, Treatment(reference='Developed'))[T.Developing]": "Developing (vs. Developed)",
        "C(Develop_Status, Treatment(reference='Developed'))[T.Mixed]": "Mixed Status (vs. Developed)",
    }

    y_col = "Atypicality_of_datasets_original_1"
    y_std = df[y_col].std() if y_col in df.columns else 1

    plot_data = []
    for model, m_type in [(model_hemi, "Hemi"), (model_dev, "Dev")]:
        for raw_var, clean_label in var_map.items():
            if raw_var in model.params:
                if (m_type == "Hemi" and "Develop" in raw_var) or (m_type == "Dev" and "Hemisphere" in raw_var):
                    continue

                coef = model.params[raw_var]
                p_val = model.pvalues[raw_var]
                stderr = model.bse[raw_var]

                if raw_var in df.columns:
                    x_std = df[raw_var].std()
                elif raw_var.startswith("C("):
                    x_std = 1
                else:
                    x_std = y_std

                std_coef = coef * (x_std / y_std)
                std_se = stderr * (x_std / y_std)
                ci_95 = 1.96 * std_se

                plot_data.append({
                    "Variable": clean_label,
                    "Coef": std_coef,
                    "CI": ci_95,
                    "P": p_val,
                    "Significance": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")),
                })

    plot_df = pd.DataFrame(plot_data).drop_duplicates(subset=["Variable"])
    plot_df = plot_df.sort_values(["Coef"], ascending=True).reset_index(drop=True)
    return plot_df


def plot_gradient_bar_chart(plot_df, output_path=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["font.size"] = 14

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    sns.set_style("white")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    cmap_pos = LinearSegmentedColormap.from_list("pos", ["#dbe9f6", "#2980b9"])
    cmap_neg = LinearSegmentedColormap.from_list("neg", ["#f6dbdb", "#c0392b"])

    max_abs_val = plot_df["Coef"].abs().max()

    bars = ax.barh(plot_df["Variable"], plot_df["Coef"],
                    xerr=plot_df["CI"],
                    error_kw={"capsize": 4, "ecolor": "#444444", "elinewidth": 1.5},
                    color=[cmap_pos(abs(row["Coef"]) / max_abs_val) if row["Coef"] > 0 else cmap_neg(abs(row["Coef"]) / max_abs_val)
                           for _, row in plot_df.iterrows()],
                    edgecolor="black", linewidth=1)

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Standardized Coefficient (with 95% CI)", fontsize=18, fontweight="bold", labelpad=15)

    for i, row in plot_df.iterrows():
        coef = row["Coef"]
        ci = row["CI"]
        sig_str = row["Significance"]

        offset = max_abs_val * 0.06
        if coef >= 0:
            text_x = coef + ci + offset
            ha = "left"
        else:
            text_x = coef - ci - offset
            ha = "right"

        ax.text(text_x, i, f"{coef:.3f}{sig_str}",
                va="center", ha=ha, fontsize=14, fontweight="bold", family="serif")

    current_xlim = ax.get_xlim()
    padding = max_abs_val * 0.3
    ax.set_xlim(current_xlim[0] - padding, current_xlim[1] + padding)

    sns.despine(left=True, bottom=False)
    plt.tight_layout(rect=[0.02, 0, 0.98, 1])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved: {output_path}")
    plt.close(fig)
    return fig


def save_results_csv(plot_df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plot_df.to_csv(filepath, index=False)
    logger.info(f"Results saved: {filepath}")


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    study_dir = get_study_dir(config)

    df = prepare_rq4_data(config)

    model_hemi, model_dev = run_rq4_regressions(df)

    plot_df = extract_rq4_results(model_hemi, model_dev, df)
    logger.info(f"Extracted {len(plot_df)} regression results")

    results_dir = os.path.join(study_dir, "results")
    save_results_csv(plot_df, os.path.join(results_dir, "rq4_regression_results.csv"))

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_gradient_bar_chart(
        plot_df,
        output_path=os.path.join(figures_dir, "rq4_gradient_bar.pdf"),
    )

    logger.info("RQ4 analysis complete.")


if __name__ == "__main__":
    main()