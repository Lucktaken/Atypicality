import os
import logging
import pandas as pd
from src.utils import load_config

logger = logging.getLogger(__name__)


class BigDataProcessor:
    """处理 100G+ 原始数据，生成约 400M 的核心中表。"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.raw_path = self.config["paths"]["nas_mount"] or self.config["paths"]["raw_data_local"]
        self.interim_path = self.config["paths"]["interim_data"]
        self.chunk_size = self.config["interim_table"]["chunk_size"]
        self.interim_filename = self.config["interim_table"]["filename"]

    def read_raw_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """分块读取大文件，返回完整 DataFrame。"""
        logger.info(f"开始读取原始数据: {filepath}")
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, **kwargs):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"读取完成，共 {len(df)} 行")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """对原始数据进行清洗、聚合，生成中表。

        请根据实际业务逻辑在此方法中实现：
        - 数据清洗（缺失值、异常值处理）
        - 字段筛选与重命名
        - 聚合计算
        """
        logger.info("开始处理数据，生成中表...")
        # TODO: 实现具体的数据处理逻辑
        interim_df = df.copy()
        logger.info(f"中表生成完成，共 {len(interim_df)} 行")
        return interim_df

    def save_interim_table(self, df: pd.DataFrame) -> str:
        """将中表保存为 parquet 格式。"""
        os.makedirs(self.interim_path, exist_ok=True)
        output_path = os.path.join(self.interim_path, self.interim_filename)
        df.to_parquet(output_path, index=False)
        logger.info(f"中表已保存至: {output_path}")
        return output_path

    def run(self):
        """完整流程：读取 → 处理 → 保存。"""
        # TODO: 根据 NAS 上实际文件名修改
        raw_file = os.path.join(self.raw_path, "raw_data.csv")
        df = self.read_raw_data(raw_file)
        interim_df = self.process(df)
        output_path = self.save_interim_table(interim_df)
        return output_path