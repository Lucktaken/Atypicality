import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_loader import load_config, load_core_table, load_q2_table
from src.regression import winsorize_df, standardize


def process_rq3_data(config: dict) -> str:
    """RQ3: 合并地理信息（半球/发展水平），生成分组回归宽表。"""
    df = load_core_table(config)
    logger.info(f"加载核心中表: {len(df)} 行")

    try:
        geo_df = load_q2_table(config, "geographic_authorlevel_middle.csv")
        df = df.merge(geo_df, on="AuthorID", how="left")
    except FileNotFoundError:
        logger.warning("未找到 geographic_authorlevel_middle.csv，跳过地理合并")

    try:
        atyp_country = load_q2_table(config, "atypicality_authorlevel_withCountry.csv")
        overlap_cols = set(df.columns) & set(atyp_country.columns) - {"AuthorID"}
        atyp_country = atyp_country.drop(columns=list(overlap_cols))
        df = df.merge(atyp_country, on="AuthorID", how="left")
    except FileNotFoundError:
        logger.warning("未找到 atypicality_authorlevel_withCountry.csv，跳过")

    y_var = "Atypicality_of_datasets_original_1"
    x_vars = ["H_index"]
    controls = ["Academic_Age", "Average_Team_Size", "Topic_Diversity"]
    group_vars = ["Hemisphere", "Develop_Status"]

    available_x = [v for v in x_vars if v in df.columns]
    available_ctrl = [c for c in controls if c in df.columns]
    available_group = [g for g in group_vars if g in df.columns]
    all_cols = [y_var] + available_x + available_ctrl + available_group
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, [y_var] + available_x + available_ctrl)
    reg_df = standardize(reg_df, available_x + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ3")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq3_regression_ready.csv")
    reg_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = load_config()
    path = process_rq3_data(config)
    print(f"RQ3 回归数据已保存至: {path}")