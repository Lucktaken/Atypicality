import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, resolve_input_path, check_file_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_study_dir(config):
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ5")


def prepare_rq5_data(config, csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, config["paths"]["interim_retention_authorlevel"])
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        sys.exit(1)

    df_final = pd.read_csv(csv_path)
    logger.info(f"Loaded retention_authorlevel: {df_final.shape}")

    df_clean = df_final[
        (df_final["Hemisphere"] != "Unknown") &
        (df_final["Develop_Status"] != "Unknown")
    ].copy()
    logger.info(f"After filtering Unknown: {df_clean.shape}")

    log_cols = ["Pub_Count", "Average_Team_Size", "H_index", "author_avg_journal_impact"]
    for col in log_cols:
        if col in df_clean.columns:
            df_clean[f"log_{col}"] = np.log1p(df_clean[col])

    mean_atyp = df_clean["Atypicality_of_datasets_original_1"].mean()
    df_clean["Atypicality_centered"] = df_clean["Atypicality_of_datasets_original_1"] - mean_atyp

    return df_clean


def run_cox_models(df_clean):
    from lifelines import CoxPHFitter

    controls = [
        "Initial_Dataset_Atyp", "Dataset_Atyp_Slope", "Dataset_Atyp_Volatility",
        "log_H_index", "Topic_Diversity", "Academic_Age",
        "log_Average_Team_Size", "log_author_avg_journal_impact",
    ]
    controls = [c for c in controls if c in df_clean.columns]

    cols_main = ["Duration", "Event", "Atypicality_centered", "Hemisphere", "Develop_Status"] + controls
    df_model = df_clean[cols_main].dropna().copy()
    df_model = pd.get_dummies(df_model, columns=["Hemisphere", "Develop_Status"], drop_first=True, dtype=int)
    logger.info(f"Cox model data: {df_model.shape}")

    cph_main = CoxPHFitter()
    cph_main.fit(df_model, duration_col="Duration", event_col="Event")
    logger.info(f"Cox main model: concordance={cph_main.concordance_index_:.4f}")

    cols_hemi = ["Duration", "Event", "Hemisphere"] + controls
    df_hemi = df_clean[cols_hemi].dropna().copy()
    df_hemi = pd.get_dummies(df_hemi, columns=["Hemisphere"])
    hemi_cols_to_drop = [c for c in df_hemi.columns if "South" in c]
    if hemi_cols_to_drop:
        df_hemi = df_hemi.drop(columns=hemi_cols_to_drop)

    cph_hemi = CoxPHFitter()
    cph_hemi.fit(df_hemi, duration_col="Duration", event_col="Event")
    logger.info(f"Cox hemisphere model: concordance={cph_hemi.concordance_index_:.4f}")

    cols_dev = ["Duration", "Event", "Develop_Status"] + controls
    df_dev = df_clean[cols_dev].dropna().copy()
    df_dev = pd.get_dummies(df_dev, columns=["Develop_Status"])
    dev_cols_to_drop = [c for c in df_dev.columns if "Developed" in c]
    if dev_cols_to_drop:
        df_dev = df_dev.drop(columns=dev_cols_to_drop)

    cph_dev = CoxPHFitter()
    cph_dev.fit(df_dev, duration_col="Duration", event_col="Event")
    logger.info(f"Cox develop_status model: concordance={cph_dev.concordance_index_:.4f}")

    return cph_main, cph_hemi, cph_dev, df_model


def extract_cox_results(cph_main, cph_hemi, cph_dev):
    rows = []
    for label, model in [("Main", cph_main), ("Hemisphere", cph_hemi), ("Develop_Status", cph_dev)]:
        summary = model.summary
        for covariate in summary.index:
            rows.append({
                "model": label,
                "covariate": covariate,
                "coef": summary.loc[covariate, "coef"],
                "exp_coef": summary.loc[covariate, "exp(coef)"],
                "se": summary.loc[covariate, "se(coef)"],
                "p": summary.loc[covariate, "p"],
                "ci_lower": summary.loc[covariate, "coef lower 95%"],
                "ci_upper": summary.loc[covariate, "coef upper 95%"],
            })
    return pd.DataFrame(rows)


def plot_km_curve(df_model, output_path=None):
    from lifelines import KaplanMeierFitter

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    TITLE_FONT = 18
    LABEL_FONT = 16
    TICK_FONT = 14
    LEGEND_FONT = 14

    df_km = df_model.copy()
    labels = ["Low Atypicality", "Medium Atypicality", "High Atypicality"]
    df_km["Atyp_Tier"] = pd.qcut(df_km["Atypicality_centered"], q=3, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    kmf = KaplanMeierFitter()

    colors = {"Low Atypicality": "#3498db", "Medium Atypicality": "#95a5a6", "High Atypicality": "#e74c3c"}

    for tier in labels:
        mask = df_km["Atyp_Tier"] == tier
        kmf.fit(df_km["Duration"][mask], event_observed=df_km["Event"][mask], label=tier)
        kmf.plot_survival_function(color=colors[tier], lw=2.5, alpha=0.9, ax=ax)

    ax.set_title("Kaplan-Meier Survival Curves by Data-Use Atypicality Tiers", fontsize=TITLE_FONT, pad=15)
    ax.set_xlabel("Duration (Years in Academia)", fontsize=LABEL_FONT)
    ax.set_ylabel("Retention Probability (Survival Rate)", fontsize=LABEL_FONT)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT)
    ax.legend(title="Atypicality Tier", title_fontsize=LEGEND_FONT, fontsize=LEGEND_FONT, loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0, 1.05)

    sns.despine()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
        logger.info(f"Figure saved: {output_path}")
    plt.close(fig)
    return fig


def save_results_csv(results_df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    logger.info(f"Results saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: 非典型性与学术生涯留存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python studies/RQ5/analysis.py                # 方案 A (默认)
  python studies/RQ5/analysis.py --plan A        # 方案 A (显式)
  python studies/RQ5/analysis.py --plan B        # 方案 B (需先运行 process_data.py)
        """)
    parser.add_argument("--plan", choices=["A", "B"], default="A",
                        help="方案选择: A=从 data/interim/ 读取, B=从 studies/RQ5/ 读取 (需先运行 process_data.py)")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    study_dir = get_study_dir(config)

    if args.plan == "B":
        csv_path = resolve_input_path(config, "RQ5", "B", "rq5_regression_ready.csv")
        if not check_file_exists(csv_path, "请先运行: python studies/RQ5/process_data.py"):
            sys.exit(1)
    else:
        csv_path = None

    df_clean = prepare_rq5_data(config, csv_path=csv_path)

    cph_main, cph_hemi, cph_dev, df_model = run_cox_models(df_clean)

    results_df = extract_cox_results(cph_main, cph_hemi, cph_dev)
    logger.info(f"Extracted {len(results_df)} Cox regression results")

    results_dir = os.path.join(study_dir, "results")
    save_results_csv(results_df, os.path.join(results_dir, "rq5_cox_results.csv"))

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_km_curve(
        df_model,
        output_path=os.path.join(figures_dir, "rq5_km_curve.jpg"),
    )

    logger.info("RQ5 analysis complete.")


if __name__ == "__main__":
    main()