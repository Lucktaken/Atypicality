"""研究问题 Q1：从中表读取数据，生成该问题专属的小表。

用法:
    python studies/research_q1/process_data.py [--config configs/config.yaml]
"""

import os
import argparse
import pandas as pd
from src.utils import load_config, setup_logging, read_interim_table


def process(config: dict) -> pd.DataFrame:
    """从中表读取数据，计算 Q1 所需特征，返回小表 DataFrame。"""
    df = read_interim_table(config)

    # TODO: 在此实现 Q1 的特征计算逻辑
    # 示例：
    # df["feature_a"] = df["col_x"] * df["col_y"]
    # df["feature_b"] = df["col_z"].rolling(7).mean()

    return df


def main():
    parser = argparse.ArgumentParser(description="研究问题 Q1 - 生成小表")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    df = process(config)

    output_dir = os.path.join(config["paths"]["studies_output"], "research_q1")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "small_table_q1.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Q1 小表已生成: {output_path}")


if __name__ == "__main__":
    main()