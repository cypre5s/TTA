"""Render the measured paired baseline and its per-sample audit links."""

import json
from pathlib import Path
from statistics import mean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ground_truth import immutable, jbytes, sha


def main():
    out = Path("reports/baseline")
    root = Path("experiments/baseline/original_v1")
    metrics = {
        arm: json.loads((out / (arm + "_metrics.json")).read_bytes())
        for arm in ["B0", "B0-E2E"]
    }
    conf = json.loads((root / "config.json").read_bytes())
    if len({m["lock_sha256"] for m in metrics.values()}) != 1:
        raise ValueError("Mismatched evaluation locks")
    names = {
        "teds": "TEDS（外表）",
        "teds_s": "TEDS-S（外表）",
        "cell_f1": "单元格 F1（含嵌套）",
        "full_exact": "完整表严格匹配率",
        "cer": "CER（含嵌套）",
        "row_accuracy": "行数正确率",
        "column_accuracy": "列数正确率",
        "span_exact": "外表跨度精确匹配率",
        "parse_ok": "严格 OTSL 可解析率",
        "cell_precision": "单元格精确率",
        "cell_recall": "单元格召回率",
        "nested_link_f1": "嵌套链接 F1",
    }

    def table(part):
        return "\n".join(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{m['metrics'][part][key]:.4f}"
                if m["metrics"][part][key] is not None
                else "无样本"
                for m in metrics.values()
            )
            + " |"
            for key, label in names.items()
        )

    refs = [
        json.loads(l)
        for l in Path("data/splits/ground_truth_v1/test.jsonl")
        .read_bytes()
        .splitlines()
    ]
    approved = {r["table_id"]: json.loads(Path(r["gt"]).read_bytes()) for r in refs}
    num_kept = sum(g["action"] == "retain_reviewed_original" for g in approved.values())
    details = {
        a: [
            json.loads(s)
            for s in (out / (a + "_samples.jsonl")).read_bytes().splitlines()
        ]
        for a in metrics
    }
    a = metrics["B0"]["metrics"]["test"]
    b = metrics["B0-E2E"]["metrics"]["test"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    keys = ["teds", "teds_s", "cell_f1", "full_exact"]
    for i, (name, m) in enumerate(metrics.items()):
        x = [k + (i - 0.5) * 0.36 for k in range(len(keys))]
        bars = ax.bar(
            x,
            [m["metrics"]["test"][k] for k in keys],
            width=0.34,
            label=name,
            color=["#167a70", "#e09537"][i],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(
        range(4), ["Outer TEDS", "Outer TEDS-S", "Hierarchical cell F1", "Full exact"]
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Held-out score")
    ax.legend(loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "baseline_test.png", dpi=160)
    plt.close(fig)
    text = f"""# Phase 4 原始模型基线

冻结 Pilot 版本为 ground_truth_v1。两个实验评估相同的 {a["tables"]} 张表、{a["pages"]} 个页面、{len(set(r["group_id"] for r in refs))} 个项目组。B0 使用原始 PP-DocLayoutV3 已保存的版面裁剪，B0-E2E 直接读取原始整页渲染。Xiaomi-OCR-0 在 {conf["gpu"]} 上以 BF16 贪心解码从头生成。

| 完整测试集 | B0 | B0-E2E |
|---|---:|---:|
{table("test")}

| 困难子集（{metrics["B0"]["metrics"]["hard_test"]["tables"]} 张） | B0 | B0-E2E |
|---|---:|---:|
{table("hard_test")}

![冻结测试集两条基线的对比](baseline_test.png)

| 系统测量 | B0 | B0-E2E |
|---|---:|---:|
| 识别总时间（秒，批内均摊） | {a["gpu_batch_seconds_allocated"]:.1f} | {b["gpu_batch_seconds_allocated"]:.1f} |
| 每页推理时间（秒） | {a["inference_seconds_per_page"]:.2f} | {b["inference_seconds_per_page"]:.2f} |
| 识别显存峰值（GiB，PyTorch 已分配） | {a["peak_allocated_gib"]:.2f} | {b["peak_allocated_gib"]:.2f} |

B0 每页时间含历史原始 PP 在对应页面的实测推理，共 {a["archived_layout_seconds"]:.1f} 秒；Xiaomi 时间计本轮实际生成，均不含模型下载、加载和 PDF 渲染。页面识别采用原始整页提示词。仅当原始 OTSL 块数与页面检测表数一致时按阅读顺序匹配，成功对齐 {b["aligned"]}/{b["tables"]} 张。歧义、缺失和严格解析失败均计入分母。

TEDS 使用固定的 IBM PubTabNet 实现，按表宏平均。单元格指标按父级路径、行列和跨度进行微平均。字符错误率计入子表，去掉排版空白。此次测试包含 {a["reference_subtables"]} 个子表；平面 OTSL 对显式子表的恢复能力由嵌套链接 F1 和完整表匹配率体现。

测试参考中有 {num_kept} 张沿用了用户批次审核通过的原始 OCR，其余参考采用源文件结构重建。这会影响基线与参考的一致程度。具有独立 PDF 或原图框的测试项共 {a["source_box_subgroup"]} 张，平均框 IoU 为 {a["source_box_mean_iou"]:.4f}。整页 25 类的完整人工框、多级表头角色和跨页逻辑表配对缺少独立标注，因此这些墙面之外的指标不计入正式成绩。

[配置与运行环境](../../experiments/baseline/original_v1/config.json) · [评分锁](../../experiments/baseline/original_v1/evaluation_lock.json) · [逐表 B0](B0_samples.jsonl) · [逐表 B0-E2E](B0-E2E_samples.jsonl)
""".replace("这些墙面之外的指标", "相应指标")
    immutable(out / "baseline_report.md", text.encode())
    immutable(
        out / "completion.json",
        jbytes(
            dict(
                status="complete",
                tables=a["tables"],
                pages=a["pages"],
                lock_sha256=metrics["B0"]["lock_sha256"],
                results={
                    arm: sha((out / (arm + "_metrics.json")).read_bytes())
                    for arm in metrics
                },
                report_sha256=sha(text.encode()),
            )
        ),
    )
    print(out / "baseline_report.md")


if __name__ == "__main__":
    main()
