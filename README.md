# TenderTable-Agent

中文招投标 PDF 的复杂表格研究工程，覆盖公开文档采集、页面渲染、版面检测、Xiaomi-OCR-0 表格识别、视觉审核、项目级数据划分、微调、消融评测和研究报告生成。

仓库只保存核心代码、配置模板和测试。原始 PDF、页面图像、表格裁片、标注、模型权重、实验输出、日志和报告均在本地生成，不随源码仓库分发。

## 主要组件

- `scripts/collect.py`、`scripts/process.py` 负责公开附件采集、元数据与 PDF 处理。
- `scripts/layout.py`、`scripts/layout_pipeline.py` 负责 PP-DocLayoutV3 版面推理。
- `scripts/cloud_ocr.py` 和 `src/xiaomi_ocr/` 负责 Xiaomi-OCR-0 推理与本地加速。
- `scripts/review.py`、`scripts/ground_truth.py` 负责审核绑定、版本化 Ground Truth 和项目隔离划分。
- `scripts/pilot_lora_train.py`、`scripts/server_layout_training.py`、`src/training/` 负责领域微调与 OTSL 修复。
- `scripts/baseline_score.py`、`scripts/ablation_score.py`、`scripts/pilot_research_report.py` 负责冻结评测、消融实验和报告生成。

## 环境与入口

```bash
cp .env.example .env
make doctor
make setup-local
make setup-layout-gpu
make resume
make status
make review-status
make audit
make test
```

本地版面运行器为 `scripts/local_paddle.sh`。Xiaomi 运行环境默认为 `../TDA/.venv/bin/python`，可通过 `PADDLE_PYTHON` 和 `XIAOMI_PYTHON` 覆盖。模型路径、批大小和推理参数位于 `configs/`。

人工决定以 JSONL 导入。每条记录需包含 `table_id`、`packet_sha256`、`reviewer_type`、`reviewer`、带时区的 `reviewed_at` 和 `decision`。批准记录还需提供 `corrected_otsl`。项目级划分、训练与最终测试只读取已冻结的版本。

```bash
make import-review DECISIONS=/path/to/decisions.jsonl
make split-reviewed VERSION=ground_truth_v1
make freeze-evaluation DATASET=/path/to/split CONFIG=/path/to/config \
  PREDICTIONS=/path/to/predictions OUTPUT=/path/to/output FREEZE=/path/to/lock
make evaluate DATASET=/path/to/split CONFIG=/path/to/config \
  PREDICTIONS=/path/to/predictions OUTPUT=/path/to/output FREEZE=/path/to/lock
```

## 数据与凭据

- `.env`、`data/`、`database/`、`models/`、`experiments/`、`logs/`、`reports/` 和缓存目录均被 Git 忽略。
- API 密钥只写入本机 `.env`。仓库中的 `.env.example` 不含密钥。
- 原始文件的访问范围、哈希和来源元数据由运行时生成并保存在本地。
- 测试集 Ground Truth 需要独立人工审核。模型辅助结果不能自行成为测试集真值。

完整研究约束见 `MASTER_TASK.md`，代理职责见 `agents/`，测试入口见 `tests/`。
