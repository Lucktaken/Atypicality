import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_loader import load_config, load_q2_table
from src.regression import winsorize_df, standardize


def process_rq4_data(config: dict) -> str:
    """RQ4: 以 Atypicality 为因变量，探究其决定因素。"""
    df = load_q2_table(config, "atypicality_authorlevel_withCountry.csv")
    logger.info(f"加载 atypicality_authorlevel_withCountry: {len(df)} 行")

    y_var = "Atypicality"
    controls = ["H_index", "Academic_Age", "Topic_Diversity", "Team_Size",
                "Avg_Citation_w/o_Self", "Unique_Dataset_Use", "Avg_Dataset_Freq"]

    available_ctrl = [c for c in controls if c in df.columns]
    all_cols = [y_var] + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, available_cols)
    reg_df = standardize(reg_df, available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ4")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq4_regression_ready.csv")
    reg_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = load_config()
    path = process_rq4_data(config)
    print(f"RQ4 回归数据已保存至: {path}")