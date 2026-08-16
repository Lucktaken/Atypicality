import os
import sys
import logging

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_authors_info, load_config, load_core_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_author_data_usage_stats(config: dict) -> pd.DataFrame:
    """从核心中表派生 author_data_usage_stats.csv（本地化，不读 DataCite）。

    paper_count = Total_Papers
    unique_data = unique_data_count
    total_data  = unique_data_count + repeated_data_usage
    """
    core = load_core_table(
        config, usecols=["AuthorID", "Total_Papers", "unique_data_count", "repeated_data_usage"]
    )
    stats = pd.DataFrame(
        {
            "AuthorID": core["AuthorID"],
            "paper_count": core["Total_Papers"],
            "unique_data": core["unique_data_count"],
            "total_data": core["unique_data_count"] + core["repeated_data_usage"],
        }
    )
    stats["Unique_Data_Paper_Ratio"] = stats["unique_data"] / stats["paper_count"].replace(0, np.nan)
    stats["Repetition_Novelty_Rate"] = 1.0 - (stats["unique_data"] / stats["total_data"].replace(0, np.nan))

    output_path = config["paths"]["interim_author_data_usage_stats"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats.to_csv(output_path, index=False)
    logger.info(f"author_data_usage_stats.csv 已保存: {output_path} (shape={stats.shape})")
    return stats


def process_rq6_data(config: dict) -> str:
    """生成 RQ6 分析数据 rq6_analysis_ready.csv。

    Word2Vec 模型（仅用于 UMAP 可视化）暂不生成。
    """
    stats = build_author_data_usage_stats(config)

    df = load_authors_info(config)
    logger.info(f"加载 authors_info: {len(df)} 行")

    core_atyp = load_core_table(config, usecols=["AuthorID", "Atypicality_of_datasets_original_1"])
    overlap = set(df.columns) & set(core_atyp.columns) - {"AuthorID"}
    if overlap:
        core_atyp = core_atyp.drop(columns=list(overlap))
    df = df.merge(core_atyp, on="AuthorID", how="left")

    overlap = set(df.columns) & set(stats.columns) - {"AuthorID"}
    stats_sub = stats.drop(columns=list(overlap)) if overlap else stats
    df = df.merge(stats_sub, on="AuthorID", how="left")

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ6")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq6_analysis_ready.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"RQ6 分析数据已保存至: {output_path} (shape={df.shape})")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq6_data(config)
    print(f"RQ6 分析数据: {path}")
