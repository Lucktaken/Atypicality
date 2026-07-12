import os
import logging
import pandas as pd
import numpy as np
from src.data_loader import load_config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Step 4: 从 authors_info + DataCite + SciSciNet 计算 author-level 特征，
    生成核心中表 authors_with_c3_c5_avg.csv"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.interim_path = self.config["paths"]["interim_data"]
        self.raw_datacite = self.config["paths"]["raw_datacite"]

    def compute_data_use_diversity(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """计算 Data-use diversity（基于 DataCite 数据集使用记录）。"""
        logger.info("计算 Data-use diversity...")
        import json
        from collections import Counter

        def calc_diversity(match_dois_str):
            if pd.isna(match_dois_str):
                return np.nan
            try:
                dois = json.loads(match_dois_str) if isinstance(match_dois_str, str) else match_dois_str
                if not dois or len(dois) == 0:
                    return np.nan
                counts = Counter(dois)
                total = sum(counts.values())
                proportions = np.array(list(counts.values())) / total
                diversity = -np.sum(proportions * np.log(proportions + 1e-12))
                return diversity
            except (json.JSONDecodeError, TypeError):
                return np.nan

        authors_df["Data_Diversity"] = authors_df["match_dois"].apply(calc_diversity)
        return authors_df

    def compute_topic_diversity(self, authors_df: pd.DataFrame, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """计算 Topic diversity（基于 paper embedding 的平均距离）。"""
        logger.info("计算 Topic diversity...")

        if "embedding" not in embeddings_df.columns:
            logger.warning("embeddings 中未找到 embedding 列，跳过 Topic Diversity")
            return authors_df

        def parse_embedding(val):
            if isinstance(val, str):
                try:
                    return np.array(json.loads(val))
                except (json.JSONDecodeError, TypeError):
                    return np.nan
            return val

        import json
        embeddings_df["emb_array"] = embeddings_df["embedding"].apply(parse_embedding)
        valid_embs = embeddings_df.dropna(subset=["emb_array"])

        author_topic_div = {}
        for author_id, group in valid_embs.groupby("AuthorID"):
            embs = np.stack(group["emb_array"].values)
            if len(embs) < 2:
                author_topic_div[author_id] = 0.0
                continue
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embs)
            avg_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            author_topic_div[author_id] = 1.0 - avg_sim

        authors_df["Topic_Diversity"] = authors_df["AuthorID"].map(author_topic_div)
        return authors_df

    def compute_academic_age(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """计算 Academic Age（从首次发表到最近发表的年数）。"""
        logger.info("计算 Academic Age...")
        if "First_Pub_Year" in authors_df.columns and "Last_Pub_Year" in authors_df.columns:
            authors_df["Academic_Age"] = authors_df["Last_Pub_Year"] - authors_df["First_Pub_Year"]
        return authors_df

    def compute_citation_metrics(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """计算 C3/C5 引用指标（需要 papers 表的引用数据）。"""
        logger.info("计算 C3/C5 引用指标...")
        # C3/C5 的计算逻辑依赖 SciSciNet 的引用数据
        # 如果 authors_df 中已有这些列则直接使用
        required_cols = ["C3", "C5"]
        for col in required_cols:
            if col not in authors_df.columns:
                logger.warning(f"未找到 {col} 列，请确保原始数据中包含引用信息")
        return authors_df

    def compute_average_team_size(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """计算 Average Team Size。"""
        logger.info("计算 Average Team Size...")
        if "Team_Size" in authors_df.columns:
            authors_df["Average_Team_Size"] = authors_df.groupby("AuthorID")["Team_Size"].transform("mean")
        return authors_df

    def run(self) -> str:
        """完整特征计算流程。"""
        from src.data_loader import load_authors_info, load_embeddings

        authors_df = load_authors_info(self.config)
        logger.info(f"加载 {len(authors_df)} 位作者")

        authors_df = self.compute_data_use_diversity(authors_df)
        authors_df = self.compute_academic_age(authors_df)
        authors_df = self.compute_citation_metrics(authors_df)
        authors_df = self.compute_average_team_size(authors_df)

        try:
            embeddings_df = load_embeddings(self.config)
            authors_df = self.compute_topic_diversity(authors_df, embeddings_df)
        except FileNotFoundError:
            logger.warning("未找到 embeddings 文件，跳过 Topic Diversity")

        os.makedirs(self.interim_path, exist_ok=True)
        output_path = os.path.join(self.interim_path, "authors_with_c3_c5_avg.csv")
        authors_df.to_csv(output_path, index=False)
        logger.info(f"核心中表已保存至: {output_path}")
        return output_path