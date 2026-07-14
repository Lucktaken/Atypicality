import os
import sys
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_core_table, load_q2_table, load_modify_table
from src.regression import winsorize_df, standardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq2_data(config: dict) -> tuple:
    df = load_core_table(config)
    logger.info(f"加载核心中表: {len(df)} 行")

    try:
        paper_atyp = load_modify_table(config, "paper_level_data_atypicality.csv")
        df = df.merge(paper_atyp, on="DOI", how="left", suffixes=("", "_paper"))
    except FileNotFoundError:
        logger.warning("未找到 paper_level_data_atypicality.csv，跳过合并")

    try:
        topic_merged = load_q2_table(config, "topic_dataset_paperlevel_merged.csv")
        df = df.merge(topic_merged, on="DOI", how="left", suffixes=("", "_topic"))
    except FileNotFoundError:
        logger.warning("未找到 topic_dataset_paperlevel_merged.csv，跳过合并")

    x_vars = ["topic_distance", "dataset_distance"]
    controls = ["H_index", "Academic_Age", "Team_Size", "Topic_Diversity"]
    y_vars = ["C3", "C5"]

    available_x = [v for v in x_vars if v in df.columns]
    available_ctrl = [c for c in controls if c in df.columns]
    available_y = [v for v in y_vars if v in df.columns]
    all_cols = available_y + available_x + available_ctrl
    available_cols = [c for c in all_cols if c in df.columns]

    reg_df = df[available_cols].dropna().copy()
    reg_df = winsorize_df(reg_df, available_cols)
    reg_df = standardize(reg_df, available_x + available_ctrl)

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ2")
    os.makedirs(output_dir, exist_ok=True)

    all_author_path = os.path.join(output_dir, "rq2_all_author_ready.csv")
    if os.path.exists(all_author_path):
        logger.info(f"全作者回归数据已存在，跳过生成: {all_author_path}")
    else:
        reg_df.to_csv(all_author_path, index=False)
        logger.info(f"全作者回归数据已保存至: {all_author_path}")

    first_author_path = None
    if "Author_Position" in df.columns:
        first_author_path = os.path.join(output_dir, "rq2_first_author_ready.csv")
        if os.path.exists(first_author_path):
            logger.info(f"第一作者回归数据已存在，跳过生成: {first_author_path}")
        else:
            first_df = df[df["Author_Position"] == "first"][available_cols].dropna().copy()
            first_df = winsorize_df(first_df, available_cols)
            first_df = standardize(first_df, available_x + available_ctrl)
            first_df.to_csv(first_author_path, index=False)
            logger.info(f"第一作者回归数据已保存至: {first_author_path}")

    return all_author_path, first_author_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    paths = process_rq2_data(config)
    print(f"RQ2 回归数据: {paths}")