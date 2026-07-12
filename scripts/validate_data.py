import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_config, setup_logging


def main():
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    core_path = config["paths"]["interim_core_table"]
    if not os.path.exists(core_path):
        logger.error(f"核心中表不存在: {core_path}")
        return

    df = pd.read_csv(core_path)
    logger.info(f"核心中表维度: {df.shape}")
    logger.info(f"列名: {list(df.columns)}")
    logger.info(f"缺失值统计:\n{df.isnull().sum()}")

    numeric_cols = df.select_dtypes(include="number").columns
    logger.info(f"数值列描述统计:\n{df[numeric_cols].describe()}")


if __name__ == "__main__":
    main()