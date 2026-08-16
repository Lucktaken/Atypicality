import os
import sys
import logging
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config
from src.q2_builders import build_atypicality_with_country

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_rq3_data(config: dict) -> str:
    """生成 RQ3 回归宽表。

    Plan A 的输入 atypicality_authorlevel_withCountry.csv 由共享 builder 真正生成
    （不再作为输入读入）。这里只做「原样写出」：预处理（缩尾/log/中心化）统一由
    `analysis.py` 的 `prepare_rq3_data` 内部完成，保证方案 A/B 结果一致。
    """
    country_path = build_atypicality_with_country(config)
    df = pd.read_csv(country_path)
    logger.info(f"加载 atypicality_authorlevel_withCountry: {len(df)} 行")

    output_dir = os.path.join(config["paths"]["studies_output"], "RQ3")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rq3_regression_ready.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"RQ3 回归数据已保存至: {output_path} (shape={df.shape})")
    return output_path


if __name__ == "__main__":
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    path = process_rq3_data(config)
    print(f"RQ3 回归数据: {path}")
