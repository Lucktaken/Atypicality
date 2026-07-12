import os
import logging
import sqlite3
import pandas as pd
from src.data_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


class BigDataProcessor:
    """Step 1-2: 从 SciSciNet + DataCite 原始数据生成 authors_info.csv"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.raw_sciscinet = self.config["paths"]["raw_sciscinet"]
        self.raw_datacite = self.config["paths"]["raw_datacite"]
        self.interim_path = self.config["paths"]["interim_data"]
        self.chunk_size = self.config["interim_table"]["chunk_size"]

    def build_sciscinet_db(self) -> str:
        """Step 1a: 将 SciSciNet TSV 加载到 SQLite/DuckDB。"""
        db_path = os.path.join(self.raw_sciscinet, "sciscinet.db")
        if os.path.exists(db_path):
            logger.info(f"数据库已存在: {db_path}")
            return db_path

        logger.info("开始构建 SciSciNet 数据库...")
        conn = sqlite3.connect(db_path)

        tsv_files = {
            "papers": "papers.tsv",
            "connections": "connections.tsv",
            "authors": "authors.tsv",
            "affiliations": "affiliations.tsv",
        }

        for table_name, tsv_file in tsv_files.items():
            tsv_path = os.path.join(self.raw_sciscinet, tsv_file)
            if not os.path.exists(tsv_path):
                logger.warning(f"文件不存在: {tsv_path}")
                continue
            logger.info(f"加载 {tsv_file} -> {table_name}...")
            for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=self.chunk_size):
                chunk.to_sql(table_name, conn, if_exists="append", index=False)

        conn.close()
        logger.info(f"数据库构建完成: {db_path}")
        return db_path

    def extract_datacite_dois(self) -> list:
        """Step 1b: 从 DataCite CSV 中提取 DOI 列表。"""
        import glob
        datacite_dir = self.raw_datacite
        all_files = sorted(glob.glob(os.path.join(datacite_dir, "*.csv")))
        logger.info(f"从 {len(all_files)} 个 DataCite 文件中提取 DOI...")

        dois = set()
        for f in all_files:
            df = pd.read_csv(f, usecols=["publication"])
            dois.update(df["publication"].dropna().tolist())

        logger.info(f"共提取 {len(dois)} 个唯一 DOI")
        return list(dois)

    def query_authors_by_dois(self, dois: list, db_path: str) -> pd.DataFrame:
        """Step 1c: 用 DOI 列表从 SciSciNet 中查询作者信息。"""
        conn = sqlite3.connect(db_path)
        batch_size = 5000
        results = []

        for i in range(0, len(dois), batch_size):
            batch = dois[i:i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            query = f"""
                SELECT p.DOI, a.Author_Name, a.AuthorID
                FROM papers p
                JOIN authors a ON p.DOI = a.DOI
                WHERE p.DOI IN ({placeholders})
            """
            chunk_df = pd.read_sql_query(query, conn, params=batch)
            results.append(chunk_df)
            if (i // batch_size) % 10 == 0:
                logger.info(f"已查询 {min(i + batch_size, len(dois))}/{len(dois)} DOI")

        conn.close()
        df = pd.concat(results, ignore_index=True)
        logger.info(f"查询完成，共 {len(df)} 条作者记录")
        return df

    def filter_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: 筛选发表 ≥10 篇且 ≥40% 出现在 DataCite 的作者。"""
        logger.info("开始筛选作者...")
        author_stats = df.groupby("AuthorID").agg(
            Total_Papers=("DOI", "count"),
            Unique_Papers=("DOI", "nunique"),
            Author_Name=("Author_Name", "first"),
        ).reset_index()

        datacite_paper_count = df.drop_duplicates("DOI").shape[0]
        author_datacite = df.groupby("AuthorID")["DOI"].nunique().reset_index()
        author_datacite.columns = ["AuthorID", "DataCite_Papers"]

        author_stats = author_stats.merge(author_datacite, on="AuthorID")
        author_stats["Percentage"] = (author_stats["DataCite_Papers"] / author_stats["Total_Papers"]) * 100

        filtered = author_stats[
            (author_stats["Total_Papers"] >= 10) & (author_stats["Percentage"] >= 40)
        ]
        logger.info(f"筛选完成：{len(filtered)} / {len(author_stats)} 位作者")
        return filtered

    def run(self) -> str:
        """完整流程：构建 DB → 提取 DOI → 查询作者 → 筛选 → 保存。"""
        db_path = self.build_sciscinet_db()
        dois = self.extract_datacite_dois()
        authors_df = self.query_authors_by_dois(dois, db_path)
        filtered_df = self.filter_authors(authors_df)

        os.makedirs(self.interim_path, exist_ok=True)
        output_path = os.path.join(self.interim_path, "authors_info.csv")
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"作者信息已保存至: {output_path}")
        return output_path