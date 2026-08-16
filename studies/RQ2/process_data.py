import os
import sys
import ast
import gc
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, load_core_table, load_paper_atypicality, load_q2_table
from src.q2_builders import _clean_doi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_emb(val):
    """解析 embedding（CSV 里存的是字符串 list）。"""
    if pd.isna(val):
        return None
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:  # noqa: BLE001
        return None


def process_rq2_data(config: dict):
    """生成 Plan A 的 paper_level_regression_ready.csv 与 Plan B 的 rq2_all_author_ready.csv。

    全作者版。一作版（paper_level_first_author_ready.csv / rq2_first_author_ready.csv）
    暂不生成（一作筛选逻辑先忽略）。
    """
    output_dir = os.path.join(config["paths"]["studies_output"], "RQ2")
    os.makedirs(output_dir, exist_ok=True)

    ready_path = config["paths"]["interim_paper_level_regression_ready"]
    rq2_all_path = os.path.join(output_dir, "rq2_all_author_ready.csv")

    if os.path.exists(ready_path) and os.path.exists(rq2_all_path):
        logger.info("RQ2 全作者数据已全部存在，跳过生成")
        return ready_path, rq2_all_path

    # Step 1: 骨架 AuthorID + paper_doi
    df = load_q2_table(config, "topic_dataset_paperlevel_merged.csv", usecols=["AuthorID", "paper_doi"])
    df["_doi"] = _clean_doi(df["paper_doi"])
    base_dois = set(df["_doi"].dropna())
    logger.info(f"骨架 {len(df)} 行, 唯一 paper_doi {len(base_dois)}")

    # Step 2: dataset_distance = paper_level_data_atypicality.paper_atypicality
    atyp = load_paper_atypicality(config, usecols=["doi", "dataset_count", "paper_atypicality"])
    atyp["_doi"] = _clean_doi(atyp["doi"])
    atyp = atyp[atyp["_doi"].isin(base_dois)].drop_duplicates("_doi")
    df = df.merge(atyp[["_doi", "dataset_count", "paper_atypicality"]], on="_doi", how="left")
    df = df.rename(columns={"paper_atypicality": "dataset_distance"})
    del atyp
    logger.info(f"dataset_distance 覆盖 {df['dataset_distance'].notna().sum()} 行")

    # Step 3: 分块读 embeddings 取 year / C3 / C5 / embeddings
    emb_path = config["paths"]["interim_embeddings"]
    chunks = []
    for chunk in pd.read_csv(emb_path, chunksize=200000, low_memory=False):
        chunk["_doi"] = _clean_doi(chunk["doi"])
        matched = chunk[chunk["_doi"].isin(base_dois)]
        if not matched.empty:
            keep = ["_doi"]
            for c in ("year", "c3", "c5", "Embeddings"):
                if c in matched.columns:
                    keep.append(c)
            chunks.append(matched[keep].copy())
    df_meta = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    df_meta = df_meta.rename(columns={"c3": "C3", "c5": "C5", "Embeddings": "embeddings"})
    df_meta = df_meta.drop_duplicates("_doi")

    df = df.merge(df_meta, on="_doi", how="left")
    del df_meta
    gc.collect()

    # Step 4: topic_distance = 作者时间线上相邻论文 embedding 的余弦距离
    df = df.dropna(subset=["year"]).sort_values(["AuthorID", "year"]).reset_index(drop=True)
    df["embeddings"] = df["embeddings"].apply(_parse_emb)
    df["prev_embeddings"] = df.groupby("AuthorID")["embeddings"].shift(1)

    def _topic_dist(row):
        e1, e2 = row["embeddings"], row["prev_embeddings"]
        if isinstance(e1, list) and isinstance(e2, list) and len(e1) == len(e2) and e1 and e2:
            try:
                return cosine(e1, e2)
            except Exception:  # noqa: BLE001
                return np.nan
        return np.nan

    df["topic_distance"] = df.apply(_topic_dist, axis=1)
    df = df.drop(columns=["prev_embeddings", "embeddings", "_doi"], errors="ignore")
    gc.collect()
    logger.info(f"topic_distance 覆盖 {df['topic_distance'].notna().sum()} 行")

    # Step 5: 作者级控制变量（核心中表）
    core = load_core_table(
        config, usecols=["AuthorID", "H_index", "Academic_Age", "Average_Team_Size", "Topic_Diversity"]
    )
    overlap = set(df.columns) & set(core.columns) - {"AuthorID"}
    if overlap:
        core = core.drop(columns=list(overlap))
    df = df.merge(core, on="AuthorID", how="left")
    del core

    # Step 6: 输出列
    keep_cols = [
        "AuthorID", "paper_doi", "year", "topic_distance", "C3", "C5",
        "H_index", "Average_Team_Size", "Academic_Age", "Topic_Diversity",
        "dataset_count", "dataset_distance",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df.to_csv(ready_path, index=False)
    logger.info(f"paper_level_regression_ready.csv 已保存: {ready_path} (shape={df.shape})")

    # Step 7: Plan B 的 rq2_all_author_ready.csv（10 列子集）
    plan_b_cols = [
        "AuthorID", "paper_doi", "C3", "C5", "topic_distance", "dataset_distance",
        "H_index", "Academic_Age", "Average_Team_Size", "Topic_Diversity",
    ]
    plan_b_cols = [c for c in plan_b_cols if c in df.columns]
    df[plan_b_cols].to_csv(rq2_all_path, index=False)
    logger.info(f"rq2_all_author_ready.csv 已保存: {rq2_all_path} (shape={df[plan_b_cols].shape})")

    return ready_path, rq2_all_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    paths = process_rq2_data(config)
    print(f"RQ2 回归数据: {paths}")
