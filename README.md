# Atypicality — 数据回归分析项目

## 项目概述

本项目是一个多研究问题的数据回归分析项目，采用**三层架构**管理数据流转，确保大文件隔离、研究问题解耦、结果可复现。

## 数据流转架构

```
┌─────────────────────────────────────────────────────────────────┐
│  基础层：100G+ 原始数据 (NAS)                                    │
│  ──→ BigDataProcessor 处理 ──→ 约 400M 中表 (data/interim/)     │
│       (scripts/build_interim_table.py)                          │
├─────────────────────────────────────────────────────────────────┤
│  派生层：每个研究问题独立                                        │
│  ──→ process_data.py 从中表计算特征 ──→ 小表 (studies/qN/)      │
├─────────────────────────────────────────────────────────────────┤
│  分析层：每个研究问题独立                                        │
│  ──→ analysis.ipynb 读取小表，回归建模 + 可视化                  │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
Atypicality/
├── .gitignore                  # 忽略所有数据文件与大文件
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── configs/
│   └── config.yaml             # 路径与参数配置（NAS路径等）
├── data/                       # 数据目录（不进 Git）
│   ├── raw/                    # 100G+ 原始数据（NAS 挂载或本地）
│   └── interim/                # 约 400M 中表（由代码生成）
├── src/                        # 核心代码库
│   ├── __init__.py
│   ├── big_data_processor.py   # 处理 100G 生成 400M 中表
│   └── utils.py                # 读取 config、通用画图等工具函数
├── scripts/
│   └── build_interim_table.py  # 执行生成 400M 中表的脚本
└── studies/                    # 核心区：按研究问题隔离
    ├── research_q1/            # 研究问题 1
    │   ├── process_data.py     # 从中表生成 Q1 小表
    │   └── analysis.ipynb      # Q1 回归分析与可视化
    └── research_q2/            # 研究问题 2
        ├── process_data.py     # 从中表生成 Q2 小表
        └── analysis.ipynb      # Q2 回归分析与可视化
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 NAS 路径

编辑 `configs/config.yaml`，将 `paths.nas_mount` 修改为你的 NAS 挂载路径：

```yaml
paths:
  nas_mount: "/mnt/nas/atypicality/raw/"   # ← 修改为实际 NAS 路径
```

如果使用本地数据，将 `paths.raw_data_local` 修改为本地原始数据目录，并将 `nas_mount` 留空。

### 3. 生成中表（400M）

```bash
python scripts/build_interim_table.py --config configs/config.yaml
```

该脚本会从 NAS（或本地）读取原始数据，处理后生成 `data/interim/interim_table.parquet`。

### 4. 生成研究问题小表

针对每个研究问题，运行对应的 `process_data.py`：

```bash
# 研究问题 Q1
python studies/research_q1/process_data.py --config configs/config.yaml

# 研究问题 Q2
python studies/research_q2/process_data.py --config configs/config.yaml
```

小表将生成在 `studies/research_qN/` 目录下。

### 5. 回归分析与可视化

打开对应研究问题的 Jupyter Notebook：

```bash
jupyter notebook studies/research_q1/analysis.ipynb
```

Notebook 中已包含读取小表、训练回归模型、可视化的模板代码。

## 新增研究问题

1. 在 `studies/` 下创建新目录，如 `research_q3/`
2. 复制 `process_data.py` 和 `analysis.ipynb` 模板
3. 在 `process_data.py` 中实现该问题的特征计算逻辑
4. 在 `analysis.ipynb` 中完成回归分析与可视化

## 注意事项

- **所有数据文件（csv/tsv/parquet/db/pkl）已被 .gitignore 忽略**，不会进入 Git
- **路径不硬编码**：所有路径统一在 `configs/config.yaml` 中配置
- **中表体积大（~400M）**，存放在团队 NAS 上，由代码动态生成到 `data/interim/`
- **小表由代码动态生成**，无需手动放置