"""RQ 共享的中间表构建器。

把 Colab 中 RQ3/RQ45 的生成逻辑提炼为可复用函数，供各 RQ 的
process_data.py 调用。只依赖基础表与 Q2 共享表（不再读 SciSciNet /
DataCite 大文件）。

    build_atypicality_with_country()  -> atypicality_authorlevel_withCountry.csv
    build_retention()                 -> retention_authorlevel.csv
"""
import os
import ast
import gc
import logging

import numpy as np
import pandas as pd

from src.data_loader import load_core_table, load_q2_table

logger = logging.getLogger(__name__)


def _clean_doi(series: pd.Series) -> pd.Series:
    """统一 DOI 清洗：转小写、去空格、去常见前缀。"""
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace("https://doi.org/", "", regex=False)
        .str.replace("http://doi.org/", "", regex=False)
        .str.replace("doi.org/", "", regex=False)
    )


def _aggregate_status(series: pd.Series) -> str:
    """把作者多机构的多值状态聚合成单一状态。"""
    vals = set(series.dropna()) - {"Unknown", ""}
    if len(vals) > 1:
        return "Mixed"
    if len(vals) == 1:
        return list(vals)[0]
    return "Unknown"


def _load_wbgapi_info():
    """加载世界银行经济表（在线），失败时返回 None。"""
    try:
        import wbgapi as wb  # noqa: PLC0415
        return wb.economy.DataFrame()[["incomeLevel", "latitude", "longitude"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"wbgapi 不可用，国家/半球列将退化为 Unknown: {exc}")
        return None


def _country_columns(geo: pd.DataFrame, countries_info):
    """由 ISO3166Code(ISO2) 派生 Latitude/Hemisphere/Develop_Status。"""
    if countries_info is None:
        geo = geo.copy()
        geo["Latitude"] = np.nan
        geo["Hemisphere"] = None
        geo["Develop_Status"] = "Unknown"
        return geo

    import pycountry  # noqa: PLC0415

    def _iso2_to_iso3(iso2):
        try:
            return pycountry.countries.get(alpha_2=str(iso2).upper()).alpha_3
        except Exception:  # noqa: BLE001
            return None

    def _lat(iso2):
        iso3 = _iso2_to_iso3(iso2)
        if iso3 is None or iso3 not in countries_info.index:
            return np.nan
        return countries_info.loc[iso3, "latitude"]

    def _hemisphere(lat):
        if pd.isna(lat):
            return None
        lat = float(lat)
        if abs(lat) <= 10:
            return "Equatorial"
        return "North" if lat > 0 else "South"

    def _dev_status(iso2):
        iso3 = _iso2_to_iso3(iso2)
        if iso3 is None or iso3 not in countries_info.index:
            return "Unknown"
        il = countries_info.loc[iso3, "incomeLevel"]
        if pd.isna(il) or il == "":
            return "Unknown"
        return "Developed" if il == "HIC" else "Developing"

    geo = geo.copy()
    geo["Latitude"] = geo["ISO3166Code"].apply(_lat)
    geo["Hemisphere"] = geo["ISO3166Code"].apply(_hemisphere)
    geo["Develop_Status"] = geo["ISO3166Code"].apply(_dev_status)
    return geo


def build_atypicality_with_country(config: dict) -> str:
    """生成 `atypicality_authorlevel_withCountry.csv`（核心中表 + 国家信息）。"""
    output_path = config["paths"]["interim_atypicality_authorlevel_withCountry"]
    if os.path.exists(output_path):
        logger.info(f"已存在，跳过生成: {output_path}")
        return output_path

    core = load_core_table(config)
    geo = load_q2_table(
        config, "geographic_authorlevel_middle.csv", usecols=["AuthorID", "ISO3166Code"]
    )
    logger.info(f"geographic_authorlevel_middle: {len(geo)} 行")

    geo = _country_columns(geo, _load_wbgapi_info())

    author_summary = (
        geo.groupby("AuthorID")
        .agg(
            ISO_List=("ISO3166Code", lambda x: [i for i in x.dropna().unique()]),
            Hemisphere=("Hemisphere", _aggregate_status),
            Develop_Status=("Develop_Status", _aggregate_status),
        )
        .reset_index()
    )

    core["AuthorID"] = core["AuthorID"].astype(str)
    author_summary["AuthorID"] = author_summary["AuthorID"].astype(str)
    result = core.merge(author_summary, on="AuthorID", how="left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"已保存: {output_path} (shape={result.shape})")
    return output_path


def _parse_match_dois(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:  # noqa: BLE001
        return []


def _compute_retention_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """对作者的长表计算留存/轨迹特征（RQ45.ipynb CELL 12）。"""

    def _slope(x, y):
        fit = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(fit) > 1 and fit["x"].nunique() > 1:
            try:
                return float(
                    np.polyfit(fit["x"].astype(float), fit["y"].astype(float), 1)[0]
                )
            except (np.linalg.LinAlgError, ValueError):
                return np.nan
        return np.nan

    def _features(group):
        group = group.sort_values("Year")
        first_year = group["Year"].min()
        last_year = group["Year"].max()
        age = group["Year"] - first_year

        slope = _slope(age, group["atypicality_score"])
        initial = group.loc[group["Year"] == first_year, "atypicality_score"].mean()
        duration = last_year - first_year
        event = 1 if last_year < 2021 else 0

        return pd.Series(
            {
                "Initial_Dataset_Atyp": initial,
                "Dataset_Atyp_Slope": slope,
                "Dataset_Atyp_Volatility": group["atypicality_score"].std() if len(group) > 1 else 0,
                "Duration": duration,
                "Event": event,
                "Last_Year": last_year,
                "Pub_Count": len(group),
            }
        )

    unique_authors = df_merged["AuthorID"].unique()
    chunk_size = 10000
    results = []
    for i in range(0, len(unique_authors), chunk_size):
        author_chunk = unique_authors[i : i + chunk_size]
        chunk_data = df_merged[df_merged["AuthorID"].isin(author_chunk)]
        results.append(
            chunk_data.groupby("AuthorID").apply(_features, include_groups=False).reset_index()
        )
    return pd.concat(results, ignore_index=True)


def build_retention(config: dict) -> str:
    """生成 `retention_authorlevel.csv`（国家信息 + 留存特征）。"""
    output_path = config["paths"]["interim_retention_authorlevel"]
    if os.path.exists(output_path):
        logger.info(f"已存在，跳过生成: {output_path}")
        return output_path

    # 1. 底表：核心中表 + 国家信息（重建，不能把 atypicality_authorlevel_withCountry 当输入）
    country_path = build_atypicality_with_country(config)
    df_author = pd.read_csv(country_path)

    df_author["match_dois"] = df_author["match_dois"].apply(_parse_match_dois)
    df_long = df_author.explode("match_dois").rename(columns={"match_dois": "doi"})
    df_long["doi_clean"] = _clean_doi(df_long["doi"])
    target_dois = set(df_long["doi_clean"].dropna().unique())

    # 2. DOI -> Year（用 papers_with_embeddings 替代 sciscinet_papers.parquet）
    emb_path = config["paths"]["interim_embeddings"]
    year_chunks = []
    for chunk in pd.read_csv(emb_path, chunksize=500000, low_memory=False, usecols=["doi", "year"]):
        chunk["doi_clean"] = _clean_doi(chunk["doi"])
        matched = chunk[chunk["doi_clean"].isin(target_dois)]
        if not matched.empty:
            year_chunks.append(matched[["doi_clean", "year"]].copy())
    df_years = pd.concat(year_chunks, ignore_index=True).drop_duplicates("doi_clean")
    df_years = df_years.rename(columns={"year": "Year"})
    del year_chunks
    gc.collect()

    df_merged = df_long[["AuthorID", "doi_clean"]].merge(df_years, on="doi_clean", how="left")

    # 3. 论文级 atypicality_score（RQ2 的共享表）
    topic = load_q2_table(
        config, "topic_dataset_paperlevel_merged.csv", usecols=["paper_doi", "atypicality_score"]
    )
    topic["doi_clean"] = _clean_doi(topic["paper_doi"])
    topic = topic.drop_duplicates("doi_clean")[["doi_clean", "atypicality_score"]]
    df_merged = df_merged.merge(topic, on="doi_clean", how="left")

    # 4. 计算留存特征
    author_features = _compute_retention_features(df_merged)
    df_author["AuthorID"] = df_author["AuthorID"].astype(str)
    author_features["AuthorID"] = author_features["AuthorID"].astype(str)
    df_final = df_author.merge(author_features, on="AuthorID", how="left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logger.info(f"已保存: {output_path} (shape={df_final.shape})")
    return output_path
