import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, resolve_input_path, check_file_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_study_dir(config):
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ1")


def winsorize_data(series, lower_percentile=1, upper_percentile=95):
    lower_bound = np.percentile(series, lower_percentile)
    upper_bound = np.percentile(series, upper_percentile)
    return np.clip(series, lower_bound, upper_bound)


def run_rq1_regressions(df):
    x_var = "Atypicality_of_datasets_original_1"
    features = [
        "Average_Team_Size",
        "Average_Institution_Citation",
        "Academic_Age",
        "Avg_Citation_Without_Self",
        x_var,
        "Topic_Diversity",
        "average_data_cite",
        "author_avg_journal_impact",
        "avg_time_period",
        "unique_data_count",
    ]
    features = [f for f in features if f in df.columns]

    all_results = {}
    for y_var in ["H_index", "Productivity", "C3", "C5"]:
        if y_var not in df.columns:
            continue

        reg_df = df[[y_var] + features].dropna().copy()
        logger.info(f"{y_var}: sample size after dropna = {len(reg_df)}")

        reg_df[y_var] = winsorize_data(reg_df[y_var], 1, 95)
        reg_df[y_var] = np.round(np.maximum(reg_df[y_var], 0)).astype(int)

        zero_prop = (reg_df[y_var] == 0).mean()
        overdispersion = reg_df[y_var].var() / max(reg_df[y_var].mean(), 1e-6)
        logger.info(f"{y_var}: zero_prop={zero_prop:.3f}, overdispersion={overdispersion:.2f}")

        for col in ["unique_data_count", "author_avg_journal_impact"]:
            if col in reg_df.columns:
                reg_df[col] = np.log1p(reg_df[col])

        scaler = StandardScaler()
        reg_df[features] = scaler.fit_transform(reg_df[features])

        X = sm.add_constant(reg_df[features])
        y = reg_df[y_var]

        best_model = None
        best_model_name = None
        best_aic = np.inf

        try:
            poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
            poisson_res = poisson_model.fit(cov_type="HC0")
            if poisson_res.converged and not pd.isna(poisson_res.pvalues).any():
                if poisson_res.aic < best_aic:
                    best_aic = poisson_res.aic
                    best_model = poisson_res
                    best_model_name = "Poisson"
                logger.info(f"{y_var} Poisson: AIC={poisson_res.aic:.2f}")
        except Exception as e:
            logger.warning(f"{y_var} Poisson failed: {e}")

        try:
            alpha_val = 1.0
            if best_model is not None and best_model_name == "Poisson":
                try:
                    aux_ols = sm.OLS((y - best_model.mu) ** 2 - y, best_model.mu ** 2).fit()
                    alpha_est = aux_ols.params.iloc[0]
                    if alpha_est > 0:
                        alpha_val = min(alpha_est, 5.0)
                except Exception:
                    pass

            glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_val))
            glm_nb_res = glm_nb.fit(cov_type="HC0", maxiter=500)
            if glm_nb_res.converged and not pd.isna(glm_nb_res.pvalues).any():
                if glm_nb_res.aic < best_aic:
                    best_aic = glm_nb_res.aic
                    best_model = glm_nb_res
                    best_model_name = "GLM_NB"
                logger.info(f"{y_var} GLM_NB (alpha={alpha_val:.4f}): AIC={glm_nb_res.aic:.2f}")
        except Exception as e:
            logger.warning(f"{y_var} GLM NB failed: {e}")

        if best_model is not None:
            coef = best_model.params.get(x_var, None)
            ci = best_model.conf_int()
            all_results[y_var] = {
                "converged": True,
                "x_var": x_var,
                "model_type": best_model_name,
                "coef": coef,
                "std_err": best_model.bse.get(x_var, None),
                "pvalue": best_model.pvalues.get(x_var, None),
                "ci_lower": ci.loc[x_var, 0] if x_var in ci.index else None,
                "ci_upper": ci.loc[x_var, 1] if x_var in ci.index else None,
                "aic": best_aic,
                "n_obs": int(best_model.nobs),
            }
            logger.info(f"{y_var} best: {best_model_name}, coef={coef:.4f}, p={best_model.pvalues.get(x_var, 0):.6f}")
        else:
            all_results[y_var] = {"converged": False, "error": "All models failed"}
            logger.warning(f"{y_var}: all models failed")

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


def plot_forest_chart(results, title="RQ1: Effect of Data Atypicality on Academic Success",
                      output_path=None, x_limits=(-0.03, 0.13)):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    FS_TITLE = 24
    FS_LABEL = 20
    FS_TICKS = 18
    FS_LEGEND = 18

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    valid_items = [(label, r) for label, r in results.items() if r.get("converged", False)]

    y_positions = np.linspace(0.8, 0.2, len(valid_items))

    for i, (label, res) in enumerate(valid_items):
        coef = res["coef"]
        ci_lower = res["ci_lower"]
        ci_upper = res["ci_upper"]
        color = colors[i % len(colors)]

        ax.errorbar(coef, y_positions[i],
                     xerr=[[coef - ci_lower], [ci_upper - coef]],
                     fmt="o", color=color, capsize=8,
                     elinewidth=2.5, markeredgewidth=2,
                     markersize=8, label=label)

    ax.axvline(x=0, color="#d62728", linestyle="--", linewidth=2, alpha=0.6)

    ax.tick_params(axis="x", labelsize=FS_TICKS)
    ax.set_xlim(x_limits)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coefficient", fontsize=FS_LABEL, labelpad=15)

    ax.legend(title="Success Metrics", title_fontsize=FS_LEGEND,
              fontsize=FS_LEGEND, loc="center right", bbox_to_anchor=(0.98, 0.5),
              frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
        logger.info(f"Figure saved: {output_path}")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="RQ1: 数据使用非典型性对学术成功的影响",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python studies/RQ1/analysis.py                # 方案 A (默认)
  python studies/RQ1/analysis.py --plan A        # 方案 A (显式)
  python studies/RQ1/analysis.py --plan B        # 方案 B (需先运行 process_data.py)
        """)
    parser.add_argument("--plan", choices=["A", "B"], default="A",
                        help="方案选择: A=从 data/interim/ 读取, B=从 studies/RQ1/ 读取 (需先运行 process_data.py)")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    study_dir = get_study_dir(config)

    if args.plan == "B":
        data_path = resolve_input_path(config, "RQ1", "B", "rq1_regression_ready.csv")
        if not check_file_exists(data_path, "请先运行: python studies/RQ1/process_data.py"):
            sys.exit(1)
    else:
        data_path = os.path.join(PROJECT_ROOT, config["paths"]["interim_core_table"])
        if not check_file_exists(data_path):
            sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded RQ1 data (Plan {args.plan}): {df.shape}")

    results = run_rq1_regressions(df)

    results_path = os.path.join(study_dir, "results", "rq1_regression_results.csv")
    save_results_csv(results, results_path)

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_forest_chart(
        results,
        title="RQ1: Effect of Data Atypicality on Academic Success",
        output_path=os.path.join(figures_dir, "rq1_forest_plot.jpg"),
    )

    logger.info("RQ1 analysis complete.")


if __name__ == "__main__":
    main()