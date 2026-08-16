import os
import sys
import logging
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config
from src.q2_builders import build_retention

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq5_data(config: dict) -> str:
    """生成 RQ5 回归宽表。

    Plan A 的输入 retention_authorlevel.csv 由共享 builder 真正生成
    （重建国家信息 + 计算留存特征）。这里只做「原样写出」：预处理统一由
    `analysis.py` 的 `prepare_rq5_data` 内部完成，保证方案 A/B 结果一致。
    """
    retention_path = build_retention(config)
    df = pd.read_csv(retention_path)
    logger.info(f"加载 retention_authorlevel: {len(df)} 行")

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ5")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq5_regression_ready.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"RQ5 回归数据已保存至: {output_path} (shape={df.shape})")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq5_data(config)
    print(f"RQ5 回归数据: {path}")
