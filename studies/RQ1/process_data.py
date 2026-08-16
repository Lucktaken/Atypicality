import os
import sys
import logging
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_core_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq1_data(config: dict) -> str:
    """生成 RQ1 回归宽表（原始列，不做缩尾/标准化）。

    与 `analysis.py` 的 `run_rq1_regressions` 使用的特征列保持一致；
    预处理（缩尾/log/标准化）统一由 analysis.py 内部完成，保证方案 A/B 结果一致。
    """
    df = load_core_table(config)
    logger.info(f"加载核心中表: {len(df)} 行")

    y_vars = ["H_index", "Productivity", "C3", "C5"]
    features = [
        "Average_Team_Size",
        "Average_Institution_Citation",
        "Academic_Age",
        "Avg_Citation_Without_Self",
        "Atypicality_of_datasets_original_1",
        "Topic_Diversity",
        "average_data_cite",
        "author_avg_journal_impact",
        "avg_time_period",
        "unique_data_count",
    ]

    keep_cols = [c for c in (y_vars + features) if c in df.columns]
    reg_df = df[keep_cols].copy()

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ1")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq1_regression_ready.csv")
    reg_df.to_csv(output_path, index=False)
    logger.info(f"RQ1 回归数据已保存至: {output_path} (shape={reg_df.shape})")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq1_data(config)
    print(f"RQ1 回归数据: {path}")
