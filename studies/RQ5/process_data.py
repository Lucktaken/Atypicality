import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_loader import load_config, load_q2_table
from src.regression import winsorize_df, standardize


def process_rq5_data(config: dict) -> str:
    """RQ5: 计算学术生涯留存指标，生成 Cox PH 回归宽表。"""
    df = load_q2_table(config, "retention_authorlevel.csv")
    logger.info(f"加载 retention_authorlevel: {len(df)} 行")

    duration_var = "Academic_Duration"
    event_var = "Retained"
    x_var = "Atypicality"
    controls = ["H_index", "Publication_Count", "Academic_Age", "Topic_Diversity"]

    available_ctrl = [c for c in controls if c in df.columns]
    all_cols = [duration_var, event_var, x_var] + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, [x_var] + available_ctrl)
    reg_df = standardize(reg_df, [x_var] + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ5")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq5_regression_ready.csv")
    reg_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = load_config()
    path = process_rq5_data(config)
    print(f"RQ5 回归数据已保存至: {path}")