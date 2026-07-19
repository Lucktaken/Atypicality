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


def _clean_doi(series: pd.Series) -> pd.Series:
    """统一清洗 DOI：去前缀、转小写、去空格"""
    return series.astype(str).str.replace("https://doi.org/", "", regex=False).str.lower().str.strip()


def process_rq2_data(config: dict) -> tuple:
    output_dir = os.path.join(config["paths"]["studies_output"], "RQ2")
    os.makedirs(output_dir, exist_ok=True)
    all_author_path = os.path.join(output_dir, "rq2_all_author_ready.csv")
    first_author_path = os.path.join(output_dir, "rq2_first_author_ready.csv")

    # 如果输出已全部存在，直接返回
    all_exists = os.path.exists(all_author_path)
    fa_exists = os.path.exists(first_author_path) if first_author_path else True
    if all_exists and fa_exists:
        logger.info(f"RQ2 回归数据已全部存在，跳过生成")
        return all_author_path, first_author_path

    # Step 1: 加载 paper-level 基础表（只取 AuthorID + paper_doi，~10MB）
    base_cols = ["AuthorID", "paper_doi"]
    df = load_q2_table(config, "topic_dataset_paperlevel_merged.csv", usecols=base_cols)
    logger.info(f"加载 paper-level 基础表: {len(df)} 行")

    # 清洗 paper_doi 用于匹配
    df["_doi_clean"] = _clean_doi(df["paper_doi"])
    base_dois = set(df["_doi_clean"].dropna())

    # Step 2: 合并 paper-level atypicality → dataset_distance
    # paper_level_data_atypicality 有 253 万行，先过滤再合并
    try:
        paper_atyp = load_modify_table(
            config, "paper_level_data_atypicality.csv",
            usecols=["doi", "paper_atypicality", "dataset_count"]
        )
        paper_atyp["_doi_clean"] = _clean_doi(paper_atyp["doi"])
        # 只保留在 base 中存在的 doi，显著减少内存
        paper_atyp = paper_atyp[paper_atyp["_doi_clean"].isin(base_dois)]
        logger.info(f"paper_level_data_atypicality 过滤后: {len(paper_atyp)} 行 (从 253 万行过滤)")
        df = df.merge(
            paper_atyp[["_doi_clean", "paper_atypicality", "dataset_count"]],
            on="_doi_clean", how="left"
        )
        df = df.rename(columns={"paper_atypicality": "dataset_distance"})
        del paper_atyp
        logger.info(f"合并 atypicality: dataset_distance 覆盖 {df['dataset_distance'].notna().sum()} 行")
    except FileNotFoundError:
        logger.warning("未找到 paper_level_data_atypicality.csv，dataset_distance 将不可用")

    # Step 3: 合并 C3/C5/topic_distance（也从 modify 表按需加载）
    try:
        existing = load_modify_table(
            config, "paper_level_regression_ready.csv",
            usecols=["paper_doi", "C3", "C5", "topic_distance", "year"]
        )
        existing["_doi_clean"] = _clean_doi(existing["paper_doi"])
        existing = existing[existing["_doi_clean"].isin(base_dois)]
        logger.info(f"paper_level_regression_ready 过滤后: {len(existing)} 行")
        df = df.merge(
            existing[["_doi_clean", "C3", "C5", "topic_distance", "year"]],
            on="_doi_clean", how="left"
        )
        del existing
        logger.info(f"合并 C3/C5/topic_distance 完成")
    except FileNotFoundError:
        logger.warning("未找到 modify/paper_level_regression_ready.csv，C3/C5/topic_distance 将不可用")

    # Step 4: 合并 author-level 控制变量
    try:
        core_cols = ["AuthorID", "H_index", "Academic_Age", "Average_Team_Size", "Topic_Diversity"]
        core_df = load_core_table(config, usecols=core_cols)
        overlap = set(df.columns) & set(core_df.columns) - {"AuthorID"}
        if overlap:
            core_df = core_df.drop(columns=list(overlap))
        df = df.merge(core_df, on="AuthorID", how="left")
        del core_df
        logger.info(f"合并核心中表控制变量完成")
    except Exception as e:
        logger.warning(f"合并核心中表失败: {e}")

    # Step 5: 清理，确定输出列
    df = df.drop(columns=["_doi_clean"], errors="ignore")

    x_vars = ["topic_distance", "dataset_distance"]
    controls = ["H_index", "Academic_Age", "Average_Team_Size", "Topic_Diversity"]
    y_vars = ["C3", "C5"]

    available_x = [v for v in x_vars if v in df.columns]
    available_ctrl = [c for c in controls if c in df.columns]
    available_y = [v for v in y_vars if v in df.columns]
    id_cols = ["AuthorID", "paper_doi"]
    available_cols = [c for c in (available_y + available_x + available_ctrl) if c in df.columns]
    keep_cols = list(dict.fromkeys(id_cols + available_cols))  # 去重保持顺序

    reg_df = df[keep_cols].dropna(subset=available_cols).copy()
    del df  # 释放内存
    logger.info(f"最终回归数据: {len(reg_df)} 行, 列: {available_cols}")

    # 全作者版本
    all_author_path = os.path.join(output_dir, "rq2_all_author_ready.csv")
    if os.path.exists(all_author_path):
        logger.info(f"全作者回归数据已存在，跳过生成: {all_author_path}")
    else:
        reg_df.to_csv(all_author_path, index=False)
        logger.info(f"全作者回归数据已保存至: {all_author_path} (shape={reg_df.shape})")

    # 第一作者版本
    first_author_path = os.path.join(output_dir, "rq2_first_author_ready.csv")
    if os.path.exists(first_author_path):
        logger.info(f"第一作者回归数据已存在，跳过生成: {first_author_path}")
    else:
        # Author_Position 不在中间表中，尝试从 modify/first_author_ready 获取已筛选的 paper_doi 列表
        try:
            fa_existing = load_modify_table(config, "paper_level_first_author_ready.csv",
                                             usecols=["paper_doi"])
            fa_existing["_doi_clean"] = _clean_doi(fa_existing["paper_doi"])
            fa_dois = set(fa_existing["_doi_clean"].dropna())
            df["_doi_clean"] = _clean_doi(df["paper_doi"])
            first_df = df[df["_doi_clean"].isin(fa_dois)][keep_cols].dropna(subset=available_cols).copy()
            df.drop(columns=["_doi_clean"], inplace=True, errors="ignore")
            first_df.to_csv(first_author_path, index=False)
            logger.info(f"第一作者回归数据已保存至: {first_author_path} (shape={first_df.shape})")
        except FileNotFoundError:
            logger.warning("未找到 modify/paper_level_first_author_ready.csv，跳过第一作者版本生成")
            first_author_path = None

    return all_author_path, first_author_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    paths = process_rq2_data(config)
    print(f"RQ2 回归数据: {paths}")