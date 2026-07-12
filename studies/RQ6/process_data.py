import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_loader import load_config, load_authors_info, load_core_table, load_q2_table
from src.regression import winsorize_df


def process_rq6_data(config: dict) -> str:
    """RQ6: 加载 Word2Vec 模型和 author stats，生成描述性分析数据。"""
    df = load_authors_info(config)
    logger.info(f"加载作者信息: {len(df)} 行")

    try:
        core_df = load_core_table(config, usecols=["AuthorID", "Atypicality_of_datasets_original_1"])
        df = df.merge(core_df, on="AuthorID", how="left")
    except Exception:
        logger.warning("未找到核心中表中的 Atypicality 列")

    try:
        stats_df = load_q2_table(config, "author_data_usage_stats.csv")
        overlap_cols = set(df.columns) & set(stats_df.columns) - {"AuthorID"}
        stats_df = stats_df.drop(columns=list(overlap_cols))
        df = df.merge(stats_df, on="AuthorID", how="left")
    except FileNotFoundError:
        logger.warning("未找到 author_data_usage_stats.csv，跳过")

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ6")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq6_analysis_ready.csv")
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = load_config()
    path = process_rq6_data(config)
    print(f"RQ6 分析数据已保存至: {path}")