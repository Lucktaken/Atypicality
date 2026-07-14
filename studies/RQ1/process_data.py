import os
import sys
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_core_table
from src.regression import winsorize_df, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq1_data(config: dict) -> str:
    df = load_core_table(config)
    logger.info(f"加载核心中表: {len(df)} 行")

    y_vars = ["H_index", "Productivity", "C3", "C5"]
    x_var = "Atypicality_of_datasets_original_1"
    controls = ["Average_Team_Size", "Avg_Citation_Without_Self", "Academic_Age", "Topic_Diversity"]

    available_y = [v for v in y_vars if v in df.columns]
    available_ctrl = [c for c in controls if c in df.columns]
    all_cols = available_y + [x_var] + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, available_cols)
    reg_df = standardize(reg_df, [x_var] + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ1")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq1_regression_ready.csv")
    if os.path.exists(output_path):
        logger.info(f"回归数据已存在，跳过生成: {output_path}")
        return output_path
    reg_df.to_csv(output_path, index=False)
    logger.info(f"RQ1 回归数据已保存至: {output_path}")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq1_data(config)
    print(f"RQ1 回归数据: {path}")