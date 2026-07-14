import os
import sys
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ3")


def winsorize_series(s, limits=(0.01, 0.99)):
    return s.clip(lower=s.quantile(limits[0]), upper=s.quantile(limits[1]))


def prepare_rq3_data(config):
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
            df["Atypicality_c"] = df[col] - df[col].mean()

    if "Academic_Age_w" in df.columns:
        df["Academic_Age_c"] = (df["Academic_Age_w"] - df["Academic_Age_w"].mean()) / df["Academic_Age_w"].std()

    if "Average_Team_Size_w" in df.columns:
        df["Team_Size_c"] = (df["Average_Team_Size_w"] - df["Average_Team_Size_w"].mean()) / df["Average_Team_Size_w"].std()

    return df


def run_rq3_regressions(df):
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

    results = {}

    y_vars = ["H_index", "Productivity", "C3", "C5"]
    y_vars = [y for y in y_vars if y in df.columns]

    for y_var in y_vars:
        df_analysis = df.copy()
        if f"log_{y_var}" not in df_analysis.columns:
            df_analysis[f"log_{y_var}"] = np.log1p(df_analysis[y_var])
        y_col = f"log_{y_var}"

        for mod_type, mod_spec in [
            ("Age", "Atypicality_c:Academic_Age_c"),
            ("Team", "Atypicality_c:Team_Size_c"),
            ("Hemi", "C(Hemisphere, Treatment(reference='South'))"),
            ("Dev", "C(Develop_Status, Treatment(reference='Developed'))"),
        ]:
            if mod_type == "Age":
                formula = f"{y_col} ~ Atypicality_c * Academic_Age_c + {variable_str}"
            elif mod_type == "Team":
                formula = f"{y_col} ~ Atypicality_c * Team_Size_c + {variable_str}"
            elif mod_type == "Hemi":
                formula = f"{y_col} ~ Atypicality_c * C(Hemisphere, Treatment(reference='South')) + {variable_str}"
            elif mod_type == "Dev":
                formula = f"{y_col} ~ Atypicality_c * C(Develop_Status, Treatment(reference='Developed')) + {variable_str}"

            try:
                model = smf.ols(formula, data=df_analysis).fit(cov_type="HC3")
                results[f"{y_var}_{mod_type}"] = model
                logger.info(f"RQ3 {y_var}_{mod_type}: R²={model.rsquared:.4f}, n={int(model.nobs)}")
            except Exception as e:
                logger.warning(f"RQ3 regression {y_var}_{mod_type} failed: {e}")

    return results


def extract_interaction_results(results):
    metrics = ["H_index", "Productivity", "C3", "C5"]
    mods = {
        "Atypicality_c:Academic_Age_c": "Academic Age",
        "Atypicality_c:Team_Size_c": "Team Size",
        "Atypicality_c:C(Hemisphere, Treatment(reference='South'))[T.North]": "North (vs. South)",
        "Atypicality_c:C(Develop_Status, Treatment(reference='Developed'))[T.Developing]": "Developing (vs. Developed)",
    }

    data = []
    for y in metrics:
        for k, v in mods.items():
            if "Academic_Age" in k:
                r = results.get(f"{y}_Age")
            elif "Team_Size" in k:
                r = results.get(f"{y}_Team")
            elif "Develop" in k:
                r = results.get(f"{y}_Dev")
            else:
                r = results.get(f"{y}_Hemi")

            if r is not None and k in r.params:
                coef = r.params[k]
                p_val = r.pvalues[k]
                ci_95 = r.bse[k] * 1.96
                data.append({"Y": y, "Mod": v, "Coef": coef, "p": p_val, "CI": ci_95})

    return pd.DataFrame(data)


def plot_butterfly_chart(plot_df, output_path=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["font.size"] = 14

    h_df = plot_df[plot_df["Y"] == "H_index"].sort_values("Mod", ascending=True).reset_index(drop=True)

    others_all = plot_df[plot_df["Y"] != "H_index"]
    idx = others_all.groupby("Mod")["Coef"].apply(lambda x: x.abs().idxmax())
    o_df = others_all.loc[idx].sort_values("Mod", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    plt.subplots_adjust(wspace=0.1, left=0.25, right=0.95)

    color_h = "#1A5276"
    color_others = "#D35400"

    axes[0].barh(h_df["Mod"], h_df["Coef"], xerr=h_df["CI"], capsize=5,
                  error_kw={"ecolor": "#333333", "elinewidth": 1.5, "alpha": 0.8},
                  color=color_h, edgecolor="black", linewidth=1.2, height=0.6)

    axes[1].barh(o_df["Mod"], o_df["Coef"], xerr=o_df["CI"], capsize=5,
                  error_kw={"ecolor": "#333333", "elinewidth": 1.5, "alpha": 0.8},
                  color=color_others, edgecolor="black", linewidth=1.2, height=0.6)

    max_coef_left = h_df["Coef"].abs().max() + h_df["CI"].max()
    axes[0].set_xlim(-max_coef_left * 1.6, max_coef_left * 1.6)

    r_max = (o_df["Coef"] + o_df["CI"]).max()
    r_min = (o_df["Coef"] - o_df["CI"]).min()
    axes[1].set_xlim(r_min - 0.2, r_max + 0.3)

    def annotate_bars(ax, df_plot, is_left):
        for i, row in df_plot.iterrows():
            patch = ax.patches[i]
            coef, p, ci, source_y = row["Coef"], row["p"], row["CI"], row["Y"]

            if p > 0.05:
                patch.set_alpha(0.5)
                patch.set_hatch("////")
            else:
                patch.set_alpha(0.85)

            label = f"{coef:.3f}*" if p <= 0.05 else f"{coef:.3f} (n.s.)"
            if not is_left:
                label += f" [{source_y}]"

            if coef >= 0:
                x_pos = coef + ci + (0.005 if is_left else 0.02)
                ha = "left"
            else:
                x_pos = coef - ci - (0.005 if is_left else 0.02)
                ha = "right"

            ax.text(x_pos, i, label, va="center", ha=ha, color="black", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    annotate_bars(axes[0], h_df, is_left=True)
    annotate_bars(axes[1], o_df, is_left=False)

    for ax in axes:
        ax.axvline(0, color="black", linewidth=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=16)

    axes[0].set_title("Career Impact (H-index)\nStandardized Coefficients", fontsize=18, color=color_h, pad=20)
    axes[1].set_title("Output Metrics (Max Effect)\nSemi-Standardized Coefficients", fontsize=18, color=color_others, pad=20)

    sig_patch = mpatches.Patch(facecolor="grey", alpha=0.85, edgecolor="black", label="Significant (p ≤ 0.05)")
    ns_patch = mpatches.Patch(facecolor="grey", alpha=0.5, hatch="////", edgecolor="black", label="Non-significant (p > 0.05)")
    err_line = plt.Line2D([0], [0], color="#333333", lw=1.5, marker="|", markersize=10, label="95% CI (Robust HC3)")
    fig.legend(handles=[sig_patch, ns_patch, err_line], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=14, frameon=False)

    plt.tight_layout()

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

    df = prepare_rq3_data(config)

    results = run_rq3_regressions(df)

    plot_df = extract_interaction_results(results)
    logger.info(f"Extracted {len(plot_df)} interaction results")

    results_dir = os.path.join(study_dir, "results")
    save_results_csv(plot_df, os.path.join(results_dir, "rq3_regression_results.csv"))

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_butterfly_chart(
        plot_df,
        output_path=os.path.join(figures_dir, "rq3_butterfly.pdf"),
    )

    logger.info("RQ3 analysis complete.")


if __name__ == "__main__":
    main()