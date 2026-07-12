import os
import yaml
import logging
import matplotlib.pyplot as plt


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """读取 YAML 配置文件，返回字典。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_path(config: dict, key: str) -> str:
    """从配置中获取路径，支持嵌套 key（用点号分隔）。

    Example:
        get_path(config, "paths.interim_data")
    """
    keys = key.split(".")
    value = config
    for k in keys:
        value = value[k]
    return value


def setup_logging(config: dict):
    """根据配置初始化 logging。"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)


def save_fig(fig: plt.Figure, filepath: str, dpi: int = 300):
    """保存 matplotlib 图表到文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def read_interim_table(config: dict) -> "pd.DataFrame":
    """读取中表（400M parquet 文件）。"""
    import pandas as pd

    interim_path = config["paths"]["interim_data"]
    filename = config["interim_table"]["filename"]
    filepath = os.path.join(interim_path, filename)
    logger = logging.getLogger(__name__)
    logger.info(f"读取中表: {filepath}")
    return pd.read_parquet(filepath)


def read_study_table(filepath: str) -> "pd.DataFrame":
    """读取研究问题的小表。"""
    import pandas as pd

    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")