import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from src.data_loader import load_config

logger = logging.getLogger(__name__)


def setup_style(config: dict = None):
    """统一可视化风格。"""
    if config is None:
        config = load_config()
    vis_cfg = config.get("visualization", {})
    plt.rcParams["font.family"] = vis_cfg.get("font_family", "sans-serif")
    plt.rcParams["font.sans-serif"] = vis_cfg.get("font_sans", ["Times New Roman", "Helvetica", "Arial"])
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = vis_cfg.get("pdf_fonttype", 42)


def save_fig(fig: plt.Figure, filepath: str, dpi: int = 300):
    """保存图表到文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图表已保存: {filepath}")


def plot_forest_chart(results_dict: dict, title: str = "Regression Results",
                      x_limits: tuple = None, figsize: tuple = (10, 6)) -> plt.Figure:
    """森林图：展示多个因变量的回归系数与置信区间。"""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    y_labels = []
    y_positions = []

    for i, (label, res) in enumerate(results_dict.items()):
        if not res.get("converged", False):
            continue
        coef = res["coef"]
        ci_lower = res["ci_lower"]
        ci_upper = res["ci_upper"]
        pval = res.get("pvalue", 1.0)

        y_pos = len(results_dict) - 1 - i
        y_positions.append(y_pos)
        y_labels.append(label)

        color = "#2166ac" if pval < 0.05 else "#cccccc"
        alpha = 1.0 if pval < 0.05 else 0.5

        ax.errorbar(coef, y_pos, xerr=[[coef - ci_lower], [ci_upper - coef]],
                     fmt="o", color=color, alpha=alpha, capsize=4, markersize=8)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)

    if x_limits:
        ax.set_xlim(x_limits)

    plt.tight_layout()
    return fig


def plot_butterfly_chart(results_dict: dict, title: str = "Comparison",
                         figsize: tuple = (12, 6)) -> plt.Figure:
    """蝴蝶图：对比两组回归结果。"""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(results_dict.keys())
    for i, label in enumerate(labels):
        res = results_dict[label]
        if not res.get("converged", False):
            continue
        coef = res["coef"]
        ci_lower = res["ci_lower"]
        ci_upper = res["ci_upper"]
        color = "#2166ac" if i == 0 else "#b2182b"

        ax.errorbar(coef, i, xerr=[[coef - ci_lower], [ci_upper - coef]],
                     fmt="s", color=color, capsize=4, markersize=8)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_km_curve(df: pd.DataFrame, duration_col: str, event_col: str,
                  group_col: str = None, title: str = "Kaplan-Meier Survival Curve",
                  figsize: tuple = (10, 6)) -> plt.Figure:
    """Kaplan-Meier 生存曲线。"""
    from lifelines import KaplanMeierFitter

    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    kmf = KaplanMeierFitter()

    if group_col and group_col in df.columns:
        for name, grouped_df in df.groupby(group_col):
            kmf.fit(grouped_df[duration_col], grouped_df[event_col], label=str(name))
            kmf.plot_survival_function(ax=ax)
    else:
        kmf.fit(df[duration_col], df[event_col])
        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel("Duration")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_binned_scatter(df: pd.DataFrame, x_var: str, y_var: str,
                        n_bins: int = 20, title: str = None,
                        figsize: tuple = (8, 6)) -> plt.Figure:
    """分箱散点图（鲁棒性检验）。"""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    df_clean = df[[x_var, y_var]].dropna()
    df_clean["bin"] = pd.qcut(df_clean[x_var], q=n_bins, labels=False, duplicates="drop")
    binned = df_clean.groupby("bin")[[x_var, y_var]].mean()

    ax.scatter(binned[x_var], binned[y_var], color="#2166ac", s=40, zorder=3, label="Binned mean")

    from numpy.polynomial import polynomial as P
    coefs = P.polyfit(df_clean[x_var], df_clean[y_var], deg=1)
    x_line = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 100)
    ax.plot(x_line, P.polyval(x_line, coefs), color="#e66101", linewidth=2, label="Linear fit")

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig