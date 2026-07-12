import os
import yaml
import logging
import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)


def load_core_table(config: dict, usecols=None) -> pd.DataFrame:
    filepath = config["paths"]["interim_core_table"]
    logging.getLogger(__name__).info(f"读取核心中表: {filepath}")
    return pd.read_csv(filepath, usecols=usecols)


def load_authors_info(config: dict, usecols=None) -> pd.DataFrame:
    filepath = config["paths"]["interim_authors_info"]
    logging.getLogger(__name__).info(f"读取作者信息: {filepath}")
    return pd.read_csv(filepath, usecols=usecols)


def load_paper_atypicality(config: dict, usecols=None) -> pd.DataFrame:
    filepath = config["paths"]["interim_paper_atypicality"]
    logging.getLogger(__name__).info(f"读取 paper-level atypicality: {filepath}")
    return pd.read_csv(filepath, usecols=usecols)


def load_embeddings(config: dict, chunksize=None) -> pd.DataFrame:
    filepath = config["paths"]["interim_embeddings"]
    logging.getLogger(__name__).info(f"读取 embeddings: {filepath}")
    if chunksize:
        return pd.read_csv(filepath, chunksize=chunksize, low_memory=False)
    return pd.read_csv(filepath, low_memory=False)


def load_q2_table(config: dict, filename: str, usecols=None) -> pd.DataFrame:
    filepath = os.path.join(config["paths"]["interim_q2"], filename)
    logging.getLogger(__name__).info(f"读取 Q2 表: {filepath}")
    return pd.read_csv(filepath, usecols=usecols)


def load_modify_table(config: dict, filename: str, usecols=None) -> pd.DataFrame:
    filepath = os.path.join(config["paths"]["interim_modify"], filename)
    logging.getLogger(__name__).info(f"读取 modify 表: {filepath}")
    return pd.read_csv(filepath, usecols=usecols)


def load_raw_datacite(config: dict, chunksize=None) -> pd.DataFrame:
    datacite_dir = config["paths"]["raw_datacite"]
    import glob
    all_files = sorted(glob.glob(os.path.join(datacite_dir, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"未在 {datacite_dir} 找到 DataCite CSV 文件")
    if chunksize:
        def generator():
            for f in all_files:
                for chunk in pd.read_csv(f, chunksize=chunksize):
                    yield chunk
        return generator()
    dfs = [pd.read_csv(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True)