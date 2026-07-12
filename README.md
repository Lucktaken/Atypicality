# Atypicality — 数据使用非典型性对学术成功的影响

## 项目概述

本项目研究数据使用的非典型性（Atypicality）如何影响学术成功，涵盖多个研究问题（Q1, RQ2-RQ6）。项目采用**三层架构**管理数据流转，确保大文件隔离、研究问题解耦、结果可复现。

## 数据流转架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  基础层：100G+ 原始数据 (SciSciNet + DataCite)                        │
│  ──→ BigDataProcessor (Step 1-2) ──→ authors_info.csv                │
│  ──→ EmbeddingGenerator (Step 3) ──→ papers_with_embeddings.csv      │
│  ──→ FeatureEngineer (Step 4) ──→ 核心中表 authors_with_c3_c5_avg.csv│
│       (~400M, 存放于 data/interim/, 不进 Git)                        │
├──────────────────────────────────────────────────────────────────────┤
│  派生层：每个研究问题独立                                             │
│  ──→ process_data.py 从中表 + Q2共享表 计算特征 ──→ 小表             │
├──────────────────────────────────────────────────────────────────────┤
│  分析层：每个研究问题独立                                             │
│  ──→ analysis.ipynb 读取小表，回归建模 + 可视化                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 研究问题总览

| RQ | 研究问题 | 因变量 | 核心自变量 | 模型 |
|----|----------|--------|------------|------|
| **Q1** | 数据使用非典型性对学术成功的影响 | H_index, Productivity, C3, C5 | Atypicality | NB / OLS / GLM |
| **RQ2** | 知识与数据的双重跨界（Paper-Level） | C3, C5 | topic_distance, dataset_distance | OLS + NB |
| **RQ3** | 地理/发展水平调节效应 | Atypicality | H_index 等 | 分组 OLS + NB |
| **RQ4** | 数据使用非典型性的决定因素 | Atypicality | H_index, Academic_Age 等 | OLS |
| **RQ5** | 非典型性与学术生涯留存 | Academic Retention | Atypicality | Cox PH + KM |
| **RQ6** | 数据使用模式的 Embedding 分析 | — (描述性) | — | Word2Vec + 聚类 |

## 目录结构

```
Atypicality/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── sciscinet/           # SciSciNet TSV 原始数据（NAS 挂载）
│   │   └── datacite/            # DataCite CSV 原始数据（NAS 挂载）
│   └── interim/
│       ├── authors_info.csv                  # Step 1-2 产出
│       ├── authors_with_c3_c5_avg.csv        # 核心中表（Step 4 产出）
│       ├── paper_level_data_atypicality.csv  # Paper-level atypicality
│       ├── papers_with_embeddings.csv        # Step 3 产出
│       ├── Q2/                               # RQ 系列共享的派生小表
│       │   ├── atypicality_authorlevel_withCountry.csv
│       │   ├── geographic_authorlevel_middle.csv
│       │   ├── retention_authorlevel.csv
│       │   ├── topic_dataset_paperlevel_merged.csv
│       │   ├── author_data_usage_stats.csv
│       │   └── dataset_word2vec.model*
│       └── modify/                           # RQ2 回归宽表
│           ├── paper_level_regression_ready.csv
│           ├── paper_level_first_author_ready.csv
│           └── ...
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # 统一数据读取（替代 Colab mount+read）
│   ├── big_data_processor.py   # Step 1-2: 原始数据 → authors_info
│   ├── feature_engineer.py     # Step 4: authors_info → 核心中表
│   ├── embedding.py            # Step 3: OpenAlex API + Sentence Transformer
│   ├── regression.py           # 通用回归工具（NB, OLS, GLM, Cox PH）
│   └── visualization.py        # 通用可视化（forest, butterfly, KM 等）
├── scripts/
│   ├── build_interim_table.py  # 执行 Step 1-4 完整流程
│   └── validate_data.py        # 数据验证与统计摘要
└── studies/
    ├── Q1/
    │   ├── process_data.py
    │   └── analysis.ipynb
    ├── RQ2/
    │   ├── process_data.py
    │   └── analysis.ipynb
    ├── RQ3/
    │   ├── process_data.py
    │   └── analysis.ipynb
    ├── RQ4/
    │   ├── process_data.py
    │   └── analysis.ipynb
    ├── RQ5/
    │   ├── process_data.py
    │   └── analysis.ipynb
    └── RQ6/
        ├── process_data.py
        └── analysis.ipynb
```

## 快速开始

### 1. Clone 仓库

```bash
git clone <repo-url> Atypicality
cd Atypicality
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置数据路径

编辑 `configs/config.yaml`：

```yaml
paths:
  # 方式一：NAS 挂载路径（存放 100G+ 原始数据）
  nas_mount: "/mnt/nas/atypicality/raw/"

  # 方式二：本地路径（如果原始数据在本地）
  raw_sciscinet: "data/raw/sciscinet/"
  raw_datacite: "data/raw/datacite/"
```

**如果已有核心中表**（`authors_with_c3_c5_avg.csv`），将其放入 `data/interim/` 即可跳过 Step 1-4。

### 4. 放置数据文件

将以下文件放入 `data/interim/` 对应位置：

| 文件 | 目标路径 | 说明 |
|------|----------|------|
| `authors_with_c3_c5_avg.csv` | `data/interim/` | 核心中表（必须） |
| `authors_info.csv` | `data/interim/` | 作者信息（RQ6 需要） |
| `paper_level_data_atypicality.csv` | `data/interim/` | Paper-level atypicality（RQ2 需要） |
| `papers_with_embeddings.csv` | `data/interim/` | Embedding 数据（RQ2 需要） |
| `Q2/` 目录下所有文件 | `data/interim/Q2/` | RQ 系列共享派生表 |
| `modify/` 目录下所有文件 | `data/interim/modify/` | RQ2 回归宽表 |

### 5. 从头生成中表（可选，需要 100G+ 原始数据）

```bash
python scripts/build_interim_table.py
```

该脚本执行 Step 1-4 完整流程：
- Step 1-2: `BigDataProcessor` 从 SciSciNet + DataCite 生成 `authors_info.csv`
- Step 3: `EmbeddingGenerator` 调用 OpenAlex API + Sentence Transformer 生成 embeddings
- Step 4: `FeatureEngineer` 计算特征生成核心中表 `authors_with_c3_c5_avg.csv`

### 6. 生成研究问题小表

```bash
python studies/Q1/process_data.py
python studies/RQ2/process_data.py
python studies/RQ3/process_data.py
python studies/RQ4/process_data.py
python studies/RQ5/process_data.py
python studies/RQ6/process_data.py
```

每个 `process_data.py` 会从核心中表和 Q2 共享表中读取数据，生成该研究问题专属的回归宽表，保存在 `studies/<RQ>/` 目录下。

### 7. 回归分析与可视化

```bash
jupyter notebook studies/Q1/analysis.ipynb
```

Notebook 中调用 `src/regression.py` 和 `src/visualization.py` 完成回归建模与图表生成，图表保存在 `studies/<RQ>/figures/` 下。

## 代码架构说明

### `src/data_loader.py` — 统一数据读取

所有数据读取通过此模块完成，替代 Colab 中的 `drive.mount` + `pd.read_csv` 模式：

```python
from src.data_loader import load_config, load_core_table, load_q2_table

config = load_config()
df = load_core_table(config)                    # 读取核心中表
q2_df = load_q2_table(config, "filename.csv")   # 读取 Q2 共享表
```

### `src/regression.py` — 通用回归工具

提供标准化的回归接口，自动处理缺失值、缩尾、标准化：

```python
from src.regression import run_ols, run_nb, run_cox_ph

result = run_ols(df, y_var="C3", x_var="Atypicality", controls=["H_index", "Academic_Age"])
result = run_nb(df, y_var="H_index", x_var="Atypicality", controls=[...])
result = run_cox_ph(df, duration_var="Duration", event_var="Event", x_var="Atypicality", controls=[...])
```

### `src/visualization.py` — 通用可视化

统一风格的图表生成，字体/颜色/尺寸从 `config.yaml` 读取：

```python
from src.visualization import plot_forest_chart, plot_km_curve, save_fig

fig = plot_forest_chart(results_dict, title="Regression Results")
save_fig(fig, "studies/Q1/figures/forest_plot.pdf")
```

## 新增研究问题

1. 在 `studies/` 下创建新目录，如 `RQ7/`
2. 创建 `process_data.py`：实现该问题的特征计算逻辑，生成回归宽表
3. 创建 `analysis.ipynb`：调用 `src/regression.py` 和 `src/visualization.py` 完成分析
4. 如需新的共享数据，放入 `data/interim/` 并在 `config.yaml` 中添加路径

## 注意事项

- **所有数据文件（csv/tsv/parquet/db/pkl/npy/model）已被 .gitignore 忽略**，不会进入 Git
- **路径不硬编码**：所有路径统一在 `configs/config.yaml` 中配置
- **核心中表体积大（~400M）**，存放在 `data/interim/`，由代码动态生成或从 NAS 复制
- **小表由 `process_data.py` 动态生成**，无需手动放置
- **Colab 工作痕迹**保留在 `colab/` 目录中（已被 .gitignore 忽略），仅供参考