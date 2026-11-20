# VSSR 表格结构恢复

PP-DocLayoutV3 和现有版面适配产生表格裁片。原始 Xiaomi-OCR-0 BF16 生成 OTSL 和可用的 OCR 文本框。VSSR 从裁片、OTSL 和可用框读取输入，并为每张表保留原始 OTSL。

| 版本 | 输入和输出 |
| --- | --- |
| VSSR-1 | RepViT、FPN、OTSL Transformer、单元格图和几何编码通过双向注意力融合。输出表级、单元格级、缺失区域和关系异常。 |
| VSSR-2 | 使用 VSSR-1 的融合特征解码单元格存在概率、边框和类型，计算行、列、邻接、跨度及无关系矩阵。 |
| VSSR-3 | 继承 VSSR-2，新增嵌套区域查询、父单元格选择及子单元格和关系解码。每个子表保留独立 ID、网格、OTSL 和 HTML。 |

`upgrade` 逐代装载已继承权重。模型通过 `RepairPipeline` 接入裁片和原始 OCR。网格重建转移现有 OCR 文本，未匹配区域返回补充 OCR 请求。保留存在冲突的候选和审核理由。部署时需校准异常置信度，当前未训练参数仅用于检查接口。

RepViT 使用 `timm/repvit_m1_0.dist_450e_in1k` 的固定公开版本。在 1024×1024 输入上输出四尺度特征，FPN 映射为 256 通道，再按配置池化以限制注意力长度。ResNet-18 可显式选择作备用入口。模型来源见 [RepViT 官方实现](https://github.com/THU-MIG/RepViT) 和 [timm 模型卡](https://huggingface.co/timm/repvit_m1_0.dist_450e_in1k)。

## 数据接口

`VSSRDataset` 接受 JSONL 或记录序列。必填 `table_id`、`image`、`raw_otsl`。可选 `image_sha256` 和 `corrected_otsl`。`geometry` 按原始 OTSL 的单元格顺序给出 `bbox` 与 `source`，未标注位置用 `null`。框格式为裁片坐标系下归一化 `x1,y1,x2,y2`。来源值为 `ground_truth`、`predicted`、`inferred` 或 `missing`。

`ocr_fragments` 接受 `text` 及可选 `bbox` 或 `grid_anchor`。`supervision` 可提供 `anomaly`、`confidence`、`cell_anomaly`、`relation_anomaly`、`missing_regions`、`cells`、`relations`、`alignment`、`nested` 和 `repair_tokens`。空标注列表表示已确认的空集，缺失或 `null` 表示未知。`cells` 按参考网格顺序，每个对象可给 `bbox`、`type`。`relations` 为同顺序的 `[Nc,Nc,5]` 多标签矩阵，负值为屏蔽项。`nested` 每项给父网格索引 `parent`、裁片内 `bbox` 和可选子 `cells` 与 `relations`。子单元格框相对其子表区域。`alignment` 需要真实标注的 `cell_index` 与归一化 `point`，`repair_tokens` 是输入位置的目标 token ID 或 `null`。

损失独立返回 `anomaly`、`cell`、`bbox`、`relation`、`alignment`、`hierarchy`、`constraint`、`repair`，总损失由显式系数求和。单元格查找用 Hungarian 匹配。未知框跳过 L1/GIoU，未知单元格和嵌套标签跳过对应监督。对齐损失只从显式标注的视觉点建正例。

## 验证和数据规划

使用 `python -m scripts.smoke_vssr_all --prepare-evidence --pilot-root ... --output ...` 挑选 8–16 张已人工批准的 Pilot 训练图及几何侧车文件。该命令读取冻结训练清单，复核裁片、审核和原始 Xiaomi 权重哈希。用这些侧车文件与 Pilot 训练数据执行真实前向。合成样本验证坏 OTSL、缺失几何、两个子表共享父单元格、八个损失的反向和一次优化，以及保存恢复。

P5 和后续结构模型数据沿用冻结的 3,500 / 750 / 750 项目分区。修复数据规划为 3,500 合成训练对、540–650 真实训练对、750 合成验证对和 115–140 真实验证对。正式规模取决于审核通过的可用标签。
