import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_loader import load_config, load_core_table
from src.regression import winsorize_df, standardize


def process_q1_data(config: dict) -> str:
    """Q1: 从核心中表提取 Q1 回归所需变量，生成回归宽表。"""
    df = load_core_table(config)
    logger.info(f"加载核心中表: {len(df)} 行")

    y_vars = ["H_index", "Productivity", "C3", "C5"]
    x_var = "Atypicality_of_datasets_original_1"
    controls = ["Team_Size", "Avg_Citation_w/o_Self", "Academic_Age", "Topic_Diversity"]

    available_y = [v for v in y_vars if v in df.columns]
    available_ctrl = [c for c in controls if c in df.columns]
    all_cols = available_y + [x_var] + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, available_cols)
    reg_df = standardize(reg_df, [x_var] + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "Q1")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "q1_regression_ready.csv")
    reg_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = load_config()
    path = process_q1_data(config)
    print(f"Q1 回归数据已保存至: {path}")