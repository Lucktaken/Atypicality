import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_config, setup_logging
from src.big_data_processor import BigDataProcessor
from src.feature_engineer import FeatureEngineer


def main():
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=== Step 1-2: 原始数据 → authors_info.csv ===")
    processor = BigDataProcessor()
    authors_info_path = processor.run()
    logger.info(f"authors_info 已生成: {authors_info_path}")

    logger.info("=== Step 4: authors_info → 核心中表 ===")
    engineer = FeatureEngineer()
    core_table_path = engineer.run()
    logger.info(f"核心中表已生成: {core_table_path}")

    logger.info("=== 全流程完成 ===")


if __name__ == "__main__":
    main()