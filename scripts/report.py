"""Refresh manifests, phase reports and an artifact-backed task ledger."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from review import verify, digest
from metadata import notice_metadata


def write(path, content):
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(content)
    tmp.replace(p)


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def local_gpu():
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        name, util, used, total = q.stdout.strip().split(", ")
        root = Path.cwd()
        active = []
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                if (proc / "cwd").resolve() == root:
                    args = (proc / "cmdline").read_bytes().split(b"\0")
                    for script in (b"scripts/cloud_ocr.py", b"scripts/layout.py"):
                        if script in args:
                            active.append(
                                {"pid": int(proc.name), "script": script.decode()}
                            )
            except OSError:
                continue
        return dict(
            name=name,
            utilization_percent=int(util),
            memory_used_mib=int(used),
            memory_total_mib=int(total),
            workers=active,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"query_error": str(exc)}


def main(audit):
    db = sqlite3.connect("database/pilot.sqlite", timeout=45)
    db.row_factory = sqlite3.Row
    docs = [dict(r) for r in db.execute("select * from documents order by sha256")]
    sources = list(map(dict, db.execute("select * from sources order by url")))
    notices = {}
    for s in sources:
        url = s["notice"]
        if url not in notices:
            page = db.execute("select body from fetched where url=?", (url,)).fetchone()
            info = {k: s[k] for k in ["project", "project_id"]}
            notices[url] = notice_metadata(page[0], info) if page else info
        s.update(notices[url])
    with db:
        db.executemany(
            "update sources set project=?,project_id=? where url=?",
            [(s["project"], s["project_id"], s["url"]) for s in sources],
        )
    bysha = defaultdict(list)
    for s in sources:
        bysha[s["sha256"]].append(s)
    tasks = []

    def task(
        phase,
        name,
        key,
        path,
        start,
        end,
        status,
        validation,
        error=None,
        retries=0,
        identity=None,
    ):
        tasks.append(
            dict(
                task_id=f"{name}-{identity or key}",
                phase_id=phase,
                task_name=name,
                input_hash=key,
                output_path=str(path),
                start_time=start,
                finish_time=end,
                status=status,
                retry_count=retries,
                error_message=error,
                validation_result=validation,
            )
        )

    page_digests = {}
    rendered = pages = 0
    page_types = Counter()
    dpi = Counter()
    skipped = []
    metadata = []
    for d in docs:
        k = d["sha256"]
        p = Path(d["path"])
        if not bysha[k] or not p.is_file() or (audit and digest(p) != k):
            raise ValueError(p)
        task(
            1,
            "download",
            k,
            p,
            d["downloaded_at"],
            d["downloaded_at"],
            "COMPLETED",
            "validated PDF and SHA256 at ingest",
        )
        mf = Path("data/rendered") / k / "manifest.json"
        expected = d["pages"]
        pages += expected
        items = json.loads(mf.read_text()) if mf.exists() else []
        if items and (
            len(items) != expected or len({x["page_id"] for x in items}) != expected
        ):
            raise ValueError(mf)
        for x in items:
            page_digests[x["page_id"]] = x["image_sha256"]
            if audit and (
                digest(Path(x["image"])) != x["image_sha256"]
                or digest(Path(x["text_layer"])) != x["text_sha256"]
            ):
                raise ValueError(x["page_id"])
            page_types[x["kind"]] += 1
            dpi[x["dpi"]] += 1
        if not items:
            skipped.append(k)
        rendered += len(items)
        finished = iso(mf.stat().st_mtime) if items else None
        start = iso((mf.parent).stat().st_ctime) if mf.parent.exists() else None
        # Use the first committed page as the first observed processing time.
        if items:
            start = iso((Path(items[0]["image"])).stat().st_mtime)
        task(
            1,
            "process",
            k,
            mf,
            start,
            finished,
            "COMPLETED" if items else "PENDING",
            f"{len(items)}/{expected} pages",
        )
        metadata.append(
            dict(**d, sources=bysha[k], render_manifest=str(mf) if items else None)
        )
    rows = verify()
    srcs = Counter(s["site"] for s in sources)
    lines = "".join(
        json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in metadata
    )
    version = (
        "pilot-"
        + hashlib.sha256("\n".join(x["sha256"] for x in docs).encode()).hexdigest()[:16]
    )
    write("data/metadata/documents.jsonl", lines)
    write(
        "data/metadata/SHA256SUMS",
        "".join(f"{x['sha256']}  {x['path']}\n" for x in docs),
    )
    write(
        "data/annotation/receipt.jsonl",
        "".join(json.dumps(x) + "\n" for x in sorted(rows, key=lambda r: r["ordinal"])),
    )
    layouts = list(Path("data/layout/predictions").glob("*.json"))
    crops = list(Path("data/tables/crops").glob("*.json"))
    predictions = list(Path("data/tables/predictions").glob("*.json"))
    expected_weight = json.loads(Path("configs/xiaomi_provenance.json").read_text())[
        "weight_sha256"
    ]
    ocr_ready = 0
    failed_structure = Counter()
    flagged_predictions = 0
    visual_structure_flags = 0
    review_fields = Counter()
    quality = Counter()
    layout_doc = set()
    pp_devices = Counter()
    ocr_devices = Counter()
    variants = Counter()
    ocr_documents = set()
    ocr_sites = Counter()
    layout_regions = {}
    checked_images = set()
    for p in layouts:
        x = json.loads(p.read_text())
        pp_devices[x.get("device", "cpu")] += 1
        layout_doc.add(x["page_id"].split("-")[0])
        if audit:
            if (
                p.stem != x["page_id"]
                or x["image_sha256"] != page_digests.get(p.stem)
                or not Path(x["preview"]).is_file()
            ):
                raise ValueError(p)
            for tid in x["tables"]:
                region = x["boxes"][int(tid.rsplit("-r", 1)[1])]
                if region["label"] != "table" or tid in layout_regions:
                    raise ValueError(tid)
                layout_regions[tid] = (
                    p.stem,
                    region["coordinate"],
                    region["polygon_points"],
                )
        task(
            2,
            "layout",
            x["image_sha256"],
            p,
            iso(p.stat().st_mtime - x["runtime_seconds"]),
            iso(p.stat().st_mtime),
            "COMPLETED",
            f"{len(x['boxes'])} regions",
            identity=x["page_id"],
        )
    if audit:
        for p in crops:
            m = json.loads(p.read_text())
            uid = m["image_sha256"]
            if (
                p.stem != m["table_id"]
                or uid in checked_images
                or digest(Path(m["image"])) != uid
                or layout_regions.get(p.stem)
                != (m["page_id"], m["bbox_px"], m["segmentation"])
                or m["document_sha256"] + f"-p{m['page_number']:04d}" != m["page_id"]
            ):
                raise ValueError(f"Crop provenance failed at {p}")
            checked_images.add(uid)
    for p in predictions:
        x = json.loads(p.read_text())
        h = p.with_suffix(".html")
        o = Path("data/otsl/raw") / (p.stem + ".otsl")
        if not (h.is_file() and o.is_file()):
            continue
        if audit:
            m = json.loads((Path("data/tables/crops") / p.name).read_text())
            if (
                x.get("weight_sha256") != expected_weight
                or x["table_id"] != p.stem
                or digest(Path(m["image"])) != x["image_sha256"]
                or o.read_text() != x["raw_otsl"]
                or x["html"] not in h.read_text()
                or not x["raw_otsl"].strip()
            ):
                raise ValueError(p)
        ocr_ready += 1
        source_doc = p.stem.split("-")[0]
        ocr_documents.add(source_doc)
        ocr_sites.update({row["site"] for row in bysha[source_doc]})
        ocr_devices[x.get("device", "NVIDIA GeForce RTX 4090")] += 1
        variants[x.get("runtime_variant", "original")] += 1
        failed_structure.update(x["structure_checks"])
        flagged_predictions += bool(x["structure_checks"])
        task(
            2,
            "ocr",
            x["image_sha256"],
            p,
            iso(p.stat().st_mtime - x["generation_seconds_batch"]),
            iso(p.stat().st_mtime),
            "COMPLETED",
            x["structure_checks"] or ["raw OTSL + HTML"],
        )
    handoff = []
    for row in rows:
        raw = Path("data/annotation/raw") / (row["table_id"] + ".json")
        ans = json.loads(raw.read_text())
        packet_path = Path("data/annotation/packets") / raw.name
        packet = json.loads(packet_path.read_text())
        handoff.append(
            dict(
                ordinal=row["ordinal"],
                table_id=row["table_id"],
                status="pending",
                document_sha256=packet["document_sha256"],
                page_number=packet["page_number"],
                bbox_px=packet["bbox_px"],
                packet_sha256=row["packet_sha256"],
                packet=str(packet_path),
                visual_review=str(raw),
                image=packet["image"],
                html=packet["html"],
                otsl=packet["raw_otsl"],
                source_urls=[s["url"] for s in bysha[packet["document_sha256"]]],
                flagged_fields=[
                    k
                    for k in (
                        "cell_errors",
                        "merge_errors",
                        "header_errors",
                        "cross_page_candidates",
                        "nested_table_candidates",
                        "text_errors",
                    )
                    if ans[k]
                ],
                structure_valid=ans["structure_valid"],
                visual_quality=ans["visual_quality"],
            )
        )
        quality[ans["visual_quality"]] += 1
        visual_structure_flags += not ans["structure_valid"]
        for f in (
            "cell_errors",
            "merge_errors",
            "header_errors",
            "cross_page_candidates",
            "nested_table_candidates",
            "text_errors",
        ):
            if ans[f]:
                review_fields[f] += 1
        task(
            2,
            "visual_review",
            row["packet_sha256"],
            raw,
            iso((Path("data/annotation/packets") / raw.name).stat().st_mtime),
            iso(raw.stat().st_mtime),
            "COMPLETED",
            row["status"],
        )
    # A later human revision closes only the exact packet that was reviewed.
    outstanding = []
    for item in handoff:
        revisions = sorted(
            (Path("data/tables/ground_truth") / item["table_id"]).glob("*.json")
        )
        latest = json.loads(revisions[-1].read_text()) if revisions else {}
        if not (
            latest.get("reviewer_type") == "human"
            and latest.get("decision") in ("approve", "reject")
            and latest.get("packet_sha256") == item["packet_sha256"]
        ):
            outstanding.append(item)
    handoff = outstanding
    handoff.sort(
        key=lambda r: (r["structure_valid"], -len(r["flagged_fields"]), r["ordinal"])
    )
    write(
        "data/annotation/human_review_queue.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in handoff),
    )
    retries = list(map(dict, db.execute("select * from retry_queue")))
    write(
        "data/metadata/retry_queue.jsonl",
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in retries),
    )
    p1 = len(docs) == 200 and not skipped
    p2 = (
        len(layouts) == pages
        and ocr_ready == len(predictions) == 1000
        and len(rows) == 1000
        and {x["ordinal"] for x in rows} == set(range(1, 1001))
        and {x["table_id"] for x in rows} == {p.stem for p in predictions}
    )
    if len(tasks) != len({t["task_id"] for t in tasks}):
        raise ValueError("Task ledger contains duplicate IDs")
    ts = iso(datetime.now().timestamp())
    phase3_receipt = Path("data/annotation/phase3_v1/completion.json")
    baseline_receipt = Path("reports/baseline/completion.json")
    phase3_done = (
        phase3_receipt.exists()
        and not handoff
        and json.loads(phase3_receipt.read_bytes()).get("status") == "complete"
    )
    phase4_done = (
        phase3_done
        and baseline_receipt.exists()
        and json.loads(baseline_receipt.read_bytes()).get("status") == "complete"
    )
    stat = dict(
        updated_at=ts,
        dataset_version=version,
        phase1="COMPLETED" if p1 else "RUNNING",
        phase2="COMPLETED" if p2 else "RUNNING",
        documents=len(docs),
        pages=pages,
        rendered=rendered,
        layout_pages=len(layouts),
        crops=len(crops),
        ocr=ocr_ready,
        ocr_documents=len(ocr_documents),
        ocr_sites=ocr_sites,
        visual_reviews=len(rows),
        pending_human=len(handoff),
        ocr_format_flags=flagged_predictions,
        visual_structure_flags=visual_structure_flags,
        visual_findings=review_fields,
        retry_queue=len(retries),
        phase3="COMPLETED" if phase3_done else "PENDING",
        phase4="COMPLETED"
        if phase4_done
        else (
            "RUNNING"
            if (
                Path("experiments/baseline/original_v1/config.json").exists()
                and phase3_done
            )
            else "PENDING"
        ),
        current_phase=4 if phase3_done else (3 if p1 and p2 else (2 if p1 else 1)),
        completed_tasks=sum(t["status"] == "COMPLETED" for t in tasks),
        failed_tasks=sum(t["status"] == "FAILED" for t in tasks),
        layout_devices=pp_devices,
        ocr_devices=ocr_devices,
        experiments=[
            str(p)
            for p in sorted(Path("reports/performance").glob("*.json"))
            if "comparison" in p.name
            or p.stem
            in {
                "pdf_workers",
                "xiaomi_extra",
                "xiaomi_deep_decode",
                "xiaomi_holdout_deep",
                "xiaomi_graph_holdout",
                "xiaomi_isolated_final",
                "xiaomi_hybrid",
                "xiaomi_tail",
                "decode_profile",
                "length_groups_walltime",
            }
        ],
        local_gpu=local_gpu(),
        best_model=None,
    )
    observed = {}
    for label, folder in (
        ("layout", "data/layout/predictions"),
        ("ocr", "data/tables/predictions"),
        ("visual_review", "data/annotation/raw"),
    ):
        stamps = [p.stat().st_mtime for p in Path(folder).glob("*.json")]
        if stamps:
            first, last = min(stamps), max(stamps)
            observed[label] = dict(
                first_output=iso(first),
                last_output=iso(last),
                elapsed_minutes=round((last - first) / 60, 2),
            )
    if docs:
        stamps = [datetime.fromisoformat(d["downloaded_at"]).timestamp() for d in docs]
        observed["collection"] = dict(
            first_output=iso(min(stamps)),
            last_output=iso(max(stamps)),
            elapsed_minutes=round((max(stamps) - min(stamps)) / 60, 2),
        )
    stat["observed_windows"] = observed
    write("database/status.json", json.dumps(stat, ensure_ascii=False, indent=2))
    write(
        "database/tasks.jsonl",
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tasks),
    )
    write(
        "reports/data_collection_report.md",
        f"""# Phase 1 数据采集 Pilot

截至 {ts}，已按内容哈希去重保存 **{len(docs)} / 200 份 PDF**，共 {pages} 页，其中 {rendered} 页已写入文档级处理清单。当前数据版本为 `{version}`。

| 发布网站 | 已关联附件 URL 数 |
| --- | ---: |
"""
        + "".join(f"| {k} | {v} |\n" for k, v in srcs.items())
        + f"""
本轮从公开采购栏目发现招标文件及采购需求等附件。原始 URL、公告链接、项目名称、原文可提取的项目编号、版本哈希、下载时间及使用范围保存在 [文件清单](../data/metadata/documents.jsonl)。公开访问情况和研究使用限制已记录，未确认再分发授权。缺失项目编号的关联有 {sum(not s["project_id"] for s in sources)} 条。

PDF 经 PyMuPDF 和 PDFium 核对页数。每页保留文本、单词及坐标、旋转矩阵、PNG 和哈希，质量侧录包含图像对比度、拉普拉斯方差和线段倾斜估计。按矢量格线密度选择渲染分辨率。200 DPI 共 {dpi[200]} 页，300 DPI 共 {dpi[300]} 页。按文本层启发式分类结果为 `{dict(page_types)}`。文本层可能来自扫描件的既有 OCR。

[输入 SHA256 清单](../data/metadata/SHA256SUMS) · [抓取重试记录](../data/metadata/retry_queue.jsonl)。候选链接发生网络或 HTTP 错误的记录为 {len(retries)} 条。""",
    )
    write(
        "reports/annotation_pilot_report.md",
        f"""# Phase 2 自动预标注 Pilot

截至 {ts}，原始 PP-DocLayoutV3 处理 {len(layouts)} 个页面，覆盖 {len(layout_doc)} 份文件，得到 {len(crops)} 张按图像哈希去重的表格裁剪。原始 Xiaomi-OCR-0 已生成 {ocr_ready} 对 OTSL / HTML，gpt-6-luna 已提交并通过 JSON Schema 校验的审核结果为 **{len(rows)} / 1,000**。

版面推理使用 PaddleX、原始 25 类和 0.25 阈值。设备页数为 `{dict(pp_devices)}`。[类别与权重哈希](../configs/layout_provenance.json)。每页保存边界框、模型多边形、类别、置信度、输出序位与预览图。OCR 使用官方表格提示词、BF16、贪心解码。按设备统计为 `{dict(ocr_devices)}`，按实现统计为 `{dict(variants)}`。云端停用后全部转为本机 RTX 4070。已识别样本来自 {len(ocr_documents)} 份文件，各来源覆盖张数为 {dict(ocr_sites)}。历史 4090 的产量由迁移前的日志核对，新写入结果逐条记录设备。每张表保存权重和输入图哈希、输出 token 数、推理批次耗时与显存峰值。原始 OTSL 格式检查标记 {flagged_predictions} 张，各类检查累计计数为 `{dict(failed_structure)}`。这些是模型输出和启发式检查，无人工真值准确率。

用户将标注模型指定为 gpt-6-luna 子代理，按 10、100、1,000 张分阶段审核。每个审核包保留图像、页码、边界框、原始 OTSL、HTML、结构检查和原始 JSON 回答。模型判为 good / medium / poor 的图像分别为 {quality["good"]} / {quality["medium"]} / {quality["poor"]} 张。按发现类型计数（每张可多项）为 `{dict(review_fields)}`。子代理判为结构有问题的有 **{visual_structure_flags} / {len(rows)} 张**。格式检查只覆盖行宽、覆盖、起始标签与生成结束，视觉审核还检查嵌套、遗漏的列和跨页边界。待人工审核 {len(handoff)} 张。

[离线图文对照预览](table_preview.html) · [人工审核队列](../data/annotation/human_review_queue.jsonl) · [审核回执](../data/annotation/receipt.jsonl) · [结构检查及预测输出](../data/tables/predictions) · [预览图](../data/layout/previews)。子代理工具未返回逐样本计费或 token 用量，这两项记为未提供。""",
    )
    from gallery import build

    build(Path("reports/table_preview.html"))
    print(json.dumps(stat, ensure_ascii=False))
    if audit and not (p1 and p2):
        raise SystemExit("Full audit pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    main(parser.parse_args().audit)
