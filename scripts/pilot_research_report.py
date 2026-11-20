"""Publish phase-specific research reports using only verified experiment outputs."""

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.ablation_score import DATA, OUT
from scripts.baseline_score import file_sha, load
from scripts.evaluate import read_split
from scripts.ground_truth import immutable, jbytes

TRAIN = Path("reports/training")
FINAL = Path("reports/final")
BASE = Path("reports/baseline")
NAMES = {
    "B0": "原始 PP、原始 Xiaomi",
    "B0-E2E": "原始 Xiaomi 整页",
    "B1": "微调 PP、原始 Xiaomi",
    "B2": "原始 PP、LoRA Xiaomi",
    "B3": "微调 PP、LoRA Xiaomi",
    "B4": "B0 加修复",
    "B5": "B1 加修复",
    "B6": "B3 加修复",
}


def write(p, title, paragraphs):
    immutable(p, ("# " + title + "\n\n" + "\n\n".join(paragraphs) + "\n").encode())


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def plot_metrics(runs):
    from io import BytesIO

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = list(runs)
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    for dx, subset, key, label, color in [
        (-0.25, "test", "teds_s", "TEDS-S (all)", "#275579"),
        (0.0, "hard_test", "teds_s", "TEDS-S (hard)", "#80adaf"),
        (0.25, "test", "cell_f1", "Cell F1 (all)", "#b97545"),
    ]:
        ax.bar(
            x + dx, [runs[a][subset][key] for a in arms], 0.24, label=label, color=color
        )
    ax.set_xticks(x, arms)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, loc="lower left")
    buffer = BytesIO()
    fig.savefig(
        buffer, format="png", dpi=160, metadata={"Software": "TenderTable-Agent"}
    )
    plt.close(fig)
    immutable(OUT / "ablation_scores.png", buffer.getvalue())


def metric_table(runs, split):
    keys = [
        "teds",
        "teds_s",
        "cell_precision",
        "cell_recall",
        "cell_f1",
        "cer",
        "full_exact",
    ]
    return table(
        [
            "配置",
            "表数",
            "TEDS",
            "TEDS-S",
            "Cell P",
            "Cell R",
            "Cell F1",
            "CER",
            "整表严格匹配",
        ],
        [
            [arm, m[split]["tables"]] + [f"{m[split][k]:.4f}" for k in keys]
            for arm, m in runs.items()
        ],
    )


def paired(rows, gold):
    groups = defaultdict(list)
    for r in rows["B6"]:
        t = r["table_id"]
        groups[gold[t]["group_id"]].append(r["teds_s"] - rows["B0"][t]["teds_s"])
    rng = random.Random(42)
    values = list(groups.values())
    estimates = []
    for _ in range(3000):
        cluster = [x for group in rng.choices(values, k=len(values)) for x in group]
        estimates.append(mean(cluster))
    estimates.sort()
    return estimates[74], estimates[2924]


def repair_report(fix, fixval):
    repairs = table(
        [
            "验证来源",
            "数目",
            "原始可解析",
            "修后可解析",
            "原始完全一致",
            "修后完全一致",
            "改动",
        ],
        [
            [
                n,
                c["total"],
                c["valid_before"],
                c["valid_after"],
                c["exact_before"],
                c["exact_after"],
                c["changed"],
            ]
            for n, c in fixval["counts"].items()
        ],
    )
    rloss = table(
        ["轮", "真实错误验证损失", "合成错误验证损失"],
        [
            [
                i["epoch"],
                f"{i['validation_loss']['real']:.4f}",
                f"{i['validation_loss']['synthetic']:.4f}",
            ]
            for i in fix["epochs"]
        ],
    )
    rtext = (
        f"修复器使用双层编码器和双层解码器，对 UTF-8 字节和六种结构标签建模。有效训练对含 {fix['train_counts']['real']} 个真实错误及 {fix['train_counts']['synthetic']} 个合成错误。"
        f"长度超过 1,536 的排除数分别为 {fix['train_counts'].get('excluded_over_length_real', 0)}、{fix['train_counts'].get('excluded_over_length_synthetic', 0)}。"
        "推理时严格解析失败才尝试单标签编辑，用已训练模型排序，并保留全部文字。"
    )
    write(
        TRAIN / "otsl_repair_report.md",
        "OTSL 结构修复 · 1k Pilot",
        [
            rtext,
            rloss,
            repairs,
            "真实验证中的可解析表也可能有错字、错列或遗漏子表，当前修复策略会保留此类输入。合成错误的改进只反映所设扰动分布。",
        ],
    )
    return repairs, rloss, rtext


def main():
    baseline_lock = load(BASE / "completion.json")
    if baseline_lock["status"] != "complete":
        raise ValueError("Baseline incomplete")
    xdirs = list(Path("models/xiaomi_lora").glob("*/training.json"))
    if len(xdirs) != 1:
        raise ValueError("One completed LoRA training run required")
    x = load(xdirs[0])
    probe = load(Path("experiments/xiaomi_adaptation/probe_r8.json"))
    layout = load(Path("experiments/layout_finetune/pilot_server/selected.json"))
    fix = load(Path("models/otsl_repair/server_v1/training.json"))
    fixval = load(Path("experiments/ablation/repair_validation.json"))
    test = read_split(DATA, "test")
    by_tid = {r["table_id"]: r for r in test}
    difficult = read_split(DATA, "hard_test")
    rawruns = {}
    runs = {}
    fp = {}
    for arm in NAMES:
        d = BASE if arm in ("B0", "B0-E2E") else OUT
        p = d / (arm + "_metrics.json")
        runs[arm] = load(p)["metrics"]
        rows = [
            json.loads(line)
            for line in (d / (arm + "_samples.jsonl")).read_bytes().splitlines()
        ]
        if {r["table_id"] for r in rows} != set(by_tid) or len(rows) != len(test):
            raise ValueError("Unpaired results " + arm)
        rawruns[arm] = rows
        fp[arm] = file_sha(p)
    layouts = {
        name: load(OUT / ("layout_" + name + ".json")) for name in ("original", "fine")
    }
    loc_table = table(
        ["PP 权重", "独立边界表数", "AP50", "AP50:95", "P@0.25", "R@0.25", "F1@0.25"],
        [
            [name, m["reference_tables"]]
            + [f"{m[k]:.4f}" for k in ("ap50", "ap50_95", "precision", "recall", "f1")]
            for name, m in layouts.items()
        ],
    )
    trials = table(
        ["试验", "Batch", "学习率", "验证框 AP", "最佳轮", "耗时（秒）"],
        [
            [
                t["name"],
                t["batch_size"],
                t["learning_rate"],
                f"{t['validation_ap']:.4f}",
                t["best_epoch"],
                f"{t['seconds']:.1f}",
            ]
            for t in layout["trials"]
        ],
    )
    layout_data = load(
        Path("data/datasets/layout/reviewed_table_pilot/source_manifest.json")
    )
    layout_text = f"官方训练配置保留 25 类输出，使用 {len(layout_data['sources']['train'])} 张训练画布和 {len(layout_data['sources']['validation'])} 张验证画布。训练图像只保留经复核表格的源文档范围，其余区域置白。试验由验证框 AP 选出 {layout['selected']}，最佳、末轮和部署权重可按训练记录中的 SHA256 核验。"
    write(
        TRAIN / "layout_finetune_report.md",
        "PP-DocLayoutV3 领域微调 · 1k Pilot",
        [
            layout_text,
            trials,
            "测试图像用相同遮罩方法单独测量定位。每张画布只有一张有独立边界的表。",
            loc_table,
            "整页文档中的其他版面区域缺少穷尽标注，25 类整页 mAP 暂缺。",
        ],
    )
    xtable = table(
        ["轮", "训练损失", "验证损失"],
        [
            [r["epoch"], f"{r['train_loss']:.5f}", f"{r['validation_loss']:.5f}"]
            for r in x["history"]
        ],
    )
    xtext = (
        f"RTX 4090 单样本 BF16 前向及反向完成，损失 {probe['loss']:.6f}，显存峰值 {probe['peak_gpu_bytes'] / 2**30:.2f} GiB。"
        f"LoRA 冻结视觉编码器，在语言侧 Q、V 投影训练 {x['trainable_parameters']:,} 个参数，秩 {x['rank']}，学习率 {x['lr']}。"
        f"训练 {x['train']} 张，验证 {x['validation']} 张，微批 {x['micro_batch']}，有效批 {x['effective_batch']}。"
        f"以验证损失选择第 {x['selected_epoch']} 轮。训练耗时 {x['seconds'] / 60:.1f} 分钟，显存峰值 {x['peak_allocated_bytes'] / 2**30:.2f} GiB。"
    )
    write(
        TRAIN / "xiaomi_finetune_feasibility.md",
        "Xiaomi-OCR-0 微调可行性 · 1k Pilot",
        [
            xtext,
            xtable,
            "原始全序列词表投影在长表上超出 24 GB。分块版本保留每个目标 token 和 EOS，只投影参与损失的隐藏状态，按 128 个 token 计算交叉熵并反向重算。",
        ],
    )
    repairs, rloss, rtext = repair_report(fix, fixval)
    plot_metrics(runs)
    b0 = {r["table_id"]: r for r in rawruns["B0"]}
    lo, hi = paired(dict(B0=b0, B6=rawruns["B6"]), by_tid)
    dt = runs["B6"]["test"]["teds_s"] - runs["B0"]["test"]["teds_s"]
    statement = (
        f"预设完整系统 B6 的 TEDS-S 相对原始级联 B0 变化 {dt * 100:+.2f} 个百分点。"
        f"按项目重采样 3,000 次，配对差的 95% 百分位区间为 [{lo * 100:+.2f}, {hi * 100:+.2f}] 个百分点。"
    )
    speed = table(
        [
            "配置",
            "行数正确率",
            "列数正确率",
            "跨度完全匹配",
            "可解析率",
            "秒/页",
            "峰值 GiB",
        ],
        [
            [a]
            + [
                f"{m['test'][k]:.4f}"
                for k in (
                    "row_accuracy",
                    "column_accuracy",
                    "span_exact",
                    "parse_ok",
                    "inference_seconds_per_page",
                    "peak_allocated_gib",
                )
            ]
            for a, m in runs.items()
        ],
    )
    mask_tids = {r["table_id"] for r in rawruns["B0"] if r["reference_iou"] is not None}
    overlap = load(Path("experiments/ablation/cohorts/content_overlap.json"))
    unseen = set(overlap["test_unseen"])
    strata = []
    for title, tids in [
        ("与训练及验证均无相同内容", unseen),
        ("独立源文档重建", mask_tids),
        (
            "含嵌套子表",
            {r["table_id"] for r in rawruns["B0"] if r["reference_subtables"]},
        ),
        ("原预标注经审核沿用", set(by_tid) - mask_tids),
    ]:
        if tids:
            vals = [
                mean(r["teds_s"] for r in rawruns[a] if r["table_id"] in tids)
                for a in ("B0", "B3", "B6")
            ]
            strata.append([title, len(tids)] + [f"{v:.4f}" for v in vals])
    complex_table = table(
        ["参考子组", "表数", "B0 TEDS-S", "B3 TEDS-S", "B6 TEDS-S"], strata
    )
    diffs = sorted(
        rawruns["B6"],
        key=lambda r: (r["teds_s"] - b0[r["table_id"]]["teds_s"], r["table_id"]),
    )
    cases = table(
        ["表格 ID", "对齐到检测框", "参考内嵌表数", "B0 TEDS-S", "B6 TEDS-S"],
        [
            [
                r["table_id"],
                "是" if r["aligned"] else "否",
                r["reference_subtables"],
                f"{b0[r['table_id']]['teds_s']:.3f}",
                f"{r['teds_s']:.3f}",
            ]
            for r in diffs[:3] + diffs[-3:]
        ],
    )
    timing = "原始版面耗时来自早期同页运行。下游识别均在 4090 上运行，批内均摊。显存列为识别时 PyTorch 已分配峰值。"
    arms = table(["配置", "模型组合"], list(NAMES.items()))
    write(
        Path("reports/evaluation/ablation_report.md"),
        "领域表格消融实验 · 1k Pilot",
        [
            statement,
            arms,
            "![消融实验的整体和困难样本指标](pilot1k/ablation_scores.png)",
            "## 完整测试集",
            metric_table(runs, "test"),
            "## 困难子集",
            metric_table(runs, "hard_test"),
            "## 结构与耗时",
            speed,
            timing,
            "## 表格定位",
            loc_table,
            "## 参考来源与嵌套表格",
            complex_table,
        ],
    )
    final_path = FINAL / "final_research_report.md"
    write(
        final_path,
        "中文采购文档表格恢复研究 · 1k Pilot",
        [
            statement,
            "## 1. 研究背景",
            "中文采购附件的评分细则、报价清单、技术参数常含跨行合并、长文本和内嵌表。实验分别测量边界定位、识别和结构修复。",
            "## 2. 相关工作",
            "PP-DocLayoutV3 提供版面区域与阅读顺序，Xiaomi-OCR-0 将裁片解码为结构序列。TEDS 源自 PubTabNet，PubTables-1M 提供表检测和结构恢复的另一种研究设定。[PaddlePaddle](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) · [Xiaomi Research](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) · [IBM PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) · [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)",
            "## 3. 数据集构建",
            f"最初从 200 份公开采购 PDF 选出 1,000 张候选。批次复核后有 999 张可用，按项目隔离为训练 {x['train']}、验证 {x['validation']}、测试 {len(test)}。困难测试 {len(difficult)} 张属于测试集。单张预测、裁片和参考带有哈希绑定。",
            "## 4. 模型结构",
            layout_text,
            rtext,
            "## 5. 训练过程",
            trials,
            xtext,
            xtable,
            rloss,
            "## 6. 实验设置",
            f"同一冻结测试集含 {len(set(v['group_id'] for v in test))} 个项目组。评测对未命中参考计零；微调检测器先对整页独立推理，再在评分时用 IoU≥0.50 一对一配对。TEDS 与 TEDS-S 评价外层网格，单元格 F1 与 CER 累计参考树的后代。",
            timing,
            "## 7. 对照实验",
            metric_table({a: runs[a] for a in ("B0", "B0-E2E")}, "test"),
            loc_table,
            "## 8. 消融实验",
            arms,
            metric_table(runs, "test"),
            speed,
            "## 9. 复杂表格专项实验",
            metric_table(runs, "hard_test"),
            complex_table,
            repairs,
            "## 10. 错误案例",
            "按完整系统相对基线的逐表 TEDS-S 差值排序，展示两端各三张。表格 ID 可在逐样本结果中定位。",
            cases,
            "## 11. 局限性",
            f"项目划分后仍有 {len(test) - len(unseen)} 张测试表的完整网格和标准化文字与训练或验证样本相同，因此另报 {len(unseen)} 张内容独立子组。测试集有 {len(test) - len(mask_tids)} 张采用经人工批量核准的原预标注，衡量新模型纠错时存在来源偏倚。"
            "检测 AP 只覆盖有独立边界的单表遮罩画布。全页 25 类穷尽标注、独立跨页逻辑表和完整多级表头角色标注缺失，相应指标暂缺。平面 OTSL 无法显式表达父子表链接。Pilot 结果与后续 5k 属不同数据版本。",
            "## 12. 后续研究计划",
            "在新项目上扩充独立参考，保持困难表与训练项目隔离。重点补充显式的子表挂接位置、表头层级和跨页链接，再评测相同配置。",
        ],
    )
    immutable(
        FINAL / "pilot1k_completion.json",
        jbytes(
            dict(
                dataset=file_sha(DATA / "manifest.json"),
                metrics=fp,
                report_sha256=file_sha(final_path),
                ablation_lock=file_sha(
                    Path("experiments/ablation/evaluation_lock.json")
                ),
                status="complete",
            )
        ),
    )
    print(final_path, flush=True)


if __name__ == "__main__":
    main()
