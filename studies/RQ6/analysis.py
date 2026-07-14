import os
import sys
import logging
import pandas as pd
import numpy as np
import ast
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_study_dir(config):
    return os.path.join(PROJECT_ROOT, config["paths"]["studies_output"], "RQ6")


def compute_author_stats(config):
    q2_path = config["paths"]["interim_q2"]
    output_path = os.path.join(PROJECT_ROOT, q2_path, "author_data_usage_stats.csv")

    if os.path.exists(output_path):
        logger.info(f"author_data_usage_stats.csv already exists, loading from {output_path}")
        return pd.read_csv(output_path)

    logger.info("Computing author data usage stats from scratch...")
    author_csv = os.path.join(PROJECT_ROOT, q2_path, "atypicality_authorlevel_withCountry.csv")
    if not os.path.exists(author_csv):
        logger.error(f"Data file not found: {author_csv}")
        sys.exit(1)

    df_author = pd.read_csv(author_csv)
    if isinstance(df_author["match_dois"].iloc[0], str):
        df_author["match_dois"] = df_author["match_dois"].apply(ast.literal_eval)

    df_author["unique_data"] = df_author["unique_data_count"]
    df_author["total_data"] = df_author["unique_data_count"] + df_author["repeated_data_usage"]
    df_author["paper_count"] = df_author["Total_Papers"]
    df_author["Unique_Data_Paper_Ratio"] = df_author["unique_data"] / df_author["paper_count"].replace(0, np.nan)
    df_author["Repetition_Novelty_Rate"] = 1 - (df_author["unique_data"] / df_author["total_data"].replace(0, np.nan))

    stats_cols = ["AuthorID", "paper_count", "unique_data", "total_data", "Unique_Data_Paper_Ratio", "Repetition_Novelty_Rate"]
    author_stats = df_author[stats_cols].copy()
    author_stats.to_csv(output_path, index=False)
    logger.info(f"Author stats saved: {output_path}")
    return author_stats


def run_word2vec_umap(config):
    q2_path = config["paths"]["interim_q2"]
    model_path = os.path.join(PROJECT_ROOT, q2_path, "dataset_word2vec.model")

    if os.path.exists(model_path):
        from gensim.models import Word2Vec
        logger.info(f"Loading existing Word2Vec model from {model_path}")
        w2v_model = Word2Vec.load(model_path)
        return w2v_model

    logger.info("Training Word2Vec model from scratch...")
    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.error("gensim not installed. Install with: pip install gensim")
        sys.exit(1)

    author_csv = os.path.join(PROJECT_ROOT, q2_path, "atypicality_authorlevel_withCountry.csv")
    if not os.path.exists(author_csv):
        logger.error(f"Data file not found: {author_csv}")
        sys.exit(1)

    df_author = pd.read_csv(author_csv)
    if isinstance(df_author["match_dois"].iloc[0], str):
        df_author["match_dois"] = df_author["match_dois"].apply(ast.literal_eval)

    sequences = df_author["match_dois"].apply(lambda x: [str(i) for i in x] if isinstance(x, list) else []).tolist()
    sequences = [seq for seq in sequences if len(seq) > 1]

    logger.info(f"Built {len(sequences)} sequences for Word2Vec training")
    w2v_model = Word2Vec(sentences=sequences, vector_size=64, window=5, min_count=2, workers=4)
    w2v_model.save(model_path)
    logger.info(f"Word2Vec model saved: {model_path}")

    del df_author
    gc.collect()

    return w2v_model


def plot_umap_projection(w2v_model, output_path=None):
    try:
        import umap.umap_ as umap
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        return None

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    words = list(w2v_model.wv.index_to_key)
    vectors = w2v_model.wv[words]
    logger.info(f"Total vectors: {len(vectors)}")

    if len(vectors) > 50000:
        logger.info(f"Sampling 50000 vectors from {len(vectors)} for UMAP")
        np.random.seed(42)
        sample_indices = np.random.choice(len(vectors), 50000, replace=False)
        vectors_sample = vectors[sample_indices]
    else:
        vectors_sample = vectors

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(vectors_sample)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.15, s=3, color="#2c3e50")
    ax.set_title("UMAP Projection of Dataset Embeddings (Full Scale)", fontsize=18, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved: {output_path}")
    plt.close(fig)
    return fig


def plot_descriptive_stats(author_stats, df_rq6, output_dir=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "Unique_Data_Paper_Ratio" in df_rq6.columns:
        axes[0, 0].hist(df_rq6["Unique_Data_Paper_Ratio"].dropna(), bins=50, color="#2980b9", alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Unique Data / Paper Ratio", fontsize=14)
        axes[0, 0].set_xlabel("Ratio")
        axes[0, 0].set_ylabel("Count")

    if "Repetition_Novelty_Rate" in df_rq6.columns:
        axes[0, 1].hist(df_rq6["Repetition_Novelty_Rate"].dropna(), bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Repetition Novelty Rate", fontsize=14)
        axes[0, 1].set_xlabel("Rate")
        axes[0, 1].set_ylabel("Count")

    if "unique_data" in df_rq6.columns:
        axes[1, 0].hist(df_rq6["unique_data"].dropna(), bins=50, color="#27ae60", alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Unique Data Count", fontsize=14)
        axes[1, 0].set_xlabel("Count")
        axes[1, 0].set_ylabel("Frequency")

    if "paper_count" in df_rq6.columns:
        axes[1, 1].hist(df_rq6["paper_count"].dropna(), bins=50, color="#8e44ad", alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Paper Count", fontsize=14)
        axes[1, 1].set_xlabel("Count")
        axes[1, 1].set_ylabel("Frequency")

    plt.suptitle("RQ6: Data Usage Pattern Descriptive Statistics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "rq6_descriptive_stats.pdf"), dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved: {output_dir}/rq6_descriptive_stats.pdf")
    plt.close(fig)
    return fig


def save_results_csv(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved: {filepath}")


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    study_dir = get_study_dir(config)

    author_stats = compute_author_stats(config)

    ready_path = os.path.join(study_dir, "rq6_analysis_ready.csv")
    if os.path.exists(ready_path):
        df_rq6 = pd.read_csv(ready_path)
    else:
        df_rq6 = author_stats.copy()
    logger.info(f"RQ6 data: {df_rq6.shape}")

    results_dir = os.path.join(study_dir, "results")
    save_results_csv(df_rq6.describe().reset_index(), os.path.join(results_dir, "rq6_descriptive_stats.csv"))

    figures_dir = os.path.join(study_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_descriptive_stats(author_stats, df_rq6, output_dir=figures_dir)

    w2v_model = run_word2vec_umap(config)
    if w2v_model is not None:
        plot_umap_projection(
            w2v_model,
            output_path=os.path.join(figures_dir, "rq6_umap_projection.pdf"),
        )

    logger.info("RQ6 analysis complete.")


if __name__ == "__main__":
    main()