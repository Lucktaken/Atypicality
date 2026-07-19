import os
import yaml
import logging
import pandas as pd


def _resolve_paths(config: dict, base_dir: str) -> dict:
    paths = config.get("paths", {})
    resolved = {}
    for key, val in paths.items():
        if isinstance(val, str) and not os.path.isabs(val):
            resolved[key] = os.path.normpath(os.path.join(base_dir, val))
        else:
            resolved[key] = val
    config["paths"] = resolved
    return config


def load_config(config_path: str = "configs/config.yaml") -> dict:
    abs_config_path = os.path.abspath(config_path)
    with open(abs_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.dirname(abs_config_path))
    config = _resolve_paths(config, base_dir)
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
    """从 data/interim/ 或其子目录 (Q2/, modify/) 读取 CSV 文件。

    会依次尝试:
    1. data/interim/<filename>
    2. data/interim/Q2/<filename>
    3. data/interim/modify/<filename>
    """
    base = config["paths"]["interim_data"]
    candidates = [
        os.path.join(base, filename),
        os.path.join(base, "Q2", filename),
        os.path.join(base, "modify", filename),
    ]
    for filepath in candidates:
        if os.path.exists(filepath):
            logging.getLogger(__name__).info(f"读取 interim 表: {filepath}")
            return pd.read_csv(filepath, usecols=usecols)
    raise FileNotFoundError(f"未找到文件 '{filename}'，尝试了: {candidates}")


def load_modify_table(config: dict, filename: str, usecols=None) -> pd.DataFrame:
    """从 data/interim/ 或其子目录 (modify/, Q2/) 读取 CSV 文件, 同 load_q2_table."""
    return load_q2_table(config, filename, usecols=usecols)


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


# ========== 方案 A/B 双模式支持 ==========

def resolve_input_path(config: dict, rq_name: str, plan: str, filename: str) -> str:
    """根据方案 (A/B) 和 RQ 名称，解析输入文件路径。

    Args:
        config: 配置字典
        rq_name: 研究问题名称，如 "RQ1"
        plan: 方案，'A' 或 'B'
        filename: 文件名，如 "rq1_regression_ready.csv"

    Returns:
        解析后的绝对文件路径

    Raises:
        ValueError: plan 不是 'A' 或 'B'
    """
    plan_upper = plan.upper()
    if plan_upper == "A":
        # 方案 A：从 data/interim/ 读取
        base = config["paths"]["interim_data"]
        return os.path.join(base, filename)
    elif plan_upper == "B":
        # 方案 B：从 studies/<RQ>/ 读取
        return os.path.join(config["paths"]["studies_output"], rq_name, filename)
    else:
        raise ValueError(f"无效方案 '{plan}'，仅支持 'A' 或 'B'")


def check_file_exists(filepath: str, context_msg: str = "") -> bool:
    """检查文件是否存在，不存在时输出可操作提示。

    Args:
        filepath: 要检查的文件路径
        context_msg: 额外的上下文信息，如"请先运行 process_data.py"

    Returns:
        True 如果文件存在，反之 False
    """
    logger = logging.getLogger(__name__)
    if os.path.exists(filepath):
        logger.info(f"文件已找到: {filepath}")
        return True
    else:
        logger.error(f"文件不存在: {filepath}")
        if context_msg:
            logger.error(f"  → {context_msg}")
        return False


def validate_required_columns(df: pd.DataFrame, required_cols: list, context_msg: str = "") -> bool:
    """检查 DataFrame 是否包含必要列，缺失时报告。

    Args:
        df: 要检查的 DataFrame
        required_cols: 必须存在的列名列表
        context_msg: 额外的上下文信息

    Returns:
        True 如果所有必要列都存在，反之 False
    """
    logger = logging.getLogger(__name__)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"缺少必要列: {missing}")
        if context_msg:
            logger.error(f"  → {context_msg}")
        return False
    logger.info(f"所有必要列校验通过 ({len(required_cols)} 列)")
    return True