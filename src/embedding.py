import os
import asyncio
import aiohttp
import logging
import pandas as pd
from src.data_loader import load_config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Step 3: 调用 OpenAlex API 获取 paper title，再用 Sentence Transformer 生成 embedding。"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.interim_path = self.config["paths"]["interim_data"]
        self.api_base_url = self.config["api"]["openalex_base_url"]
        self.api_email = self.config["api"]["openalex_email"]

    async def fetch_title(self, session: aiohttp.ClientSession, doi: str) -> dict:
        """异步获取单篇论文的 title。"""
        headers = {"User-Agent": f"mailto:{self.api_email}"}
        try:
            async with session.get(self.api_base_url + doi, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {"DOI": doi, "Title": data.get("title", None)}
                else:
                    return {"DOI": doi, "Title": None}
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return {"DOI": doi, "Title": None}

    async def batch_fetch_titles(self, dois: list, batch_size: int = 50) -> list:
        """批量异步获取 titles。"""
        results = []
        for i in range(0, len(dois), batch_size):
            batch = dois[i:i + batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_title(session, doi) for doi in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
            if (i // batch_size) % 20 == 0:
                logger.info(f"已获取 {min(i + batch_size, len(dois))}/{len(dois)} titles")
        return results

    def fetch_titles(self, dois: list) -> pd.DataFrame:
        """同步接口：获取所有 titles。"""
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.run(self.batch_fetch_titles(dois))
        return pd.DataFrame(results)

    def generate_embeddings(self, titles_df: pd.DataFrame) -> pd.DataFrame:
        """用 Sentence Transformer 生成 embedding。"""
        from sentence_transformers import SentenceTransformer

        logger.info("加载 Sentence Transformer 模型...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        valid_mask = titles_df["Title"].notna()
        valid_titles = titles_df.loc[valid_mask, "Title"].tolist()

        logger.info(f"为 {len(valid_titles)} 篇论文生成 embedding...")
        embeddings = model.encode(valid_titles, show_progress_bar=True)

        import numpy as np
        import json
        titles_df.loc[valid_mask, "embedding"] = [json.dumps(emb.tolist()) for emb in embeddings]
        return titles_df

    def run(self) -> str:
        """完整流程：获取 titles → 生成 embeddings → 保存。"""
        from src.data_loader import load_authors_info

        authors_df = load_authors_info(self.config)
        dois = authors_df["DOI"].unique().tolist() if "DOI" in authors_df.columns else []
        logger.info(f"共 {len(dois)} 个 DOI 需要获取 title")

        titles_df = self.fetch_titles(dois)
        titles_df = self.generate_embeddings(titles_df)

        os.makedirs(self.interim_path, exist_ok=True)
        output_path = os.path.join(self.interim_path, "papers_with_embeddings.csv")
        titles_df.to_csv(output_path, index=False)
        logger.info(f"Embeddings 已保存至: {output_path}")
        return output_path