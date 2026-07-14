import os
import sys
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_q2_table
from src.regression import winsorize_df, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq5_data(config: dict) -> str:
    df = load_q2_table(config, "retention_authorlevel.csv")
    logger.info(f"加载 retention_authorlevel: {len(df)} 行")

    duration_var = "Duration"
    event_var = "Event"
    x_var = "Atypicality_of_datasets_original_1"
    controls = ["H_index", "Pub_Count", "Academic_Age", "Topic_Diversity"]

    available_ctrl = [c for c in controls if c in df.columns]
    all_cols = [duration_var, event_var, x_var] + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, [x_var] + available_ctrl)
    reg_df = standardize(reg_df, [x_var] + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ5")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq5_regression_ready.csv")
    if os.path.exists(output_path):
        logger.info(f"回归数据已存在，跳过生成: {output_path}")
        return output_path
    reg_df.to_csv(output_path, index=False)
    logger.info(f"RQ5 回归数据已保存至: {output_path}")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq5_data(config)
    print(f"RQ5 回归数据: {path}")