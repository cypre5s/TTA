"""Prepare independent adjacent-page link review and evaluate it on frozen B0 outputs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from scripts.evaluate import read_split
from scripts.expansion_gallery import preview
from scripts.ground_truth import immutable, jbytes, sha
from src.cross_page.candidates import candidate
from src.otsl.grid import parse

THRESHOLD = 0.72
DECISIONS = {"same_table", "different_tables", "uncertain"}
LINKER = Path("src/cross_page/candidates.py")


def load(path):
    return json.loads(path.read_bytes())


def evidence(row):
    packet_path = Path(row["packet"])
    data = packet_path.read_bytes()
    packet = json.loads(data)
    if sha(data) != row["packet_sha256"] or packet["table_id"] != row["table_id"]:
        raise ValueError("Changed crop metadata")
    reference = Path(row["reference"])
    body = reference.read_bytes()
    if sha(body) != row["reference_sha256"]:
        raise ValueError("Changed reference")
    page_alias = Path("data/rendered", packet["document_sha256"], packet["page_id"] + ".json")
    page_path = Path(json.loads(body).get("source_page", page_alias))
    page = load(page_path)
    if page["page_id"] != packet["page_id"]:
        raise ValueError("Incorrect source page")
    return dict(
        table_id=row["table_id"], document_sha256=packet["document_sha256"],
        page_number=packet["page_number"], page_size_px=page["size_px"],
        bbox_px=packet["bbox_px"], image=row["image"],
        image_sha256=row["image_sha256"], packet_sha256=row["packet_sha256"],
    )


def select_pairs(tables, limit):
    by_doc = defaultdict(lambda: defaultdict(list))
    for table in tables.values():
        by_doc[table["document_sha256"]][table["page_number"]].append(table)
    boundary, others = [], []
    for pages in by_doc.values():
        for number, rows in sorted(pages.items()):
            for first in rows:
                for second in pages.get(number + 1, []):
                    pair = [first["table_id"], second["table_id"]]
                    key = sha(jbytes(pair))
                    near_edges = (
                        first["bbox_px"][3] / first["page_size_px"][1] > 0.65
                        and second["bbox_px"][1] / second["page_size_px"][1] < 0.35
                    )
                    (boundary if near_edges else others).append((key, pair))
    boundary.sort()
    others.sort()
    n_boundary = min(len(boundary), (2 * limit + 2) // 3)
    n_other = min(len(others), limit - n_boundary)
    n_boundary = min(len(boundary), limit - n_other)
    picked = boundary[:n_boundary] + others[:n_other]
    return len(boundary) + len(others), picked


GALLERY = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>跨页表格配对复核</title>
<style>
body{margin:20px auto;max-width:1450px;padding:0 16px;font:16px system-ui;color:#182532}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,input,select{font:inherit;padding:5px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}section{overflow:auto;border:1px solid #bbc7d0;padding:8px}img{width:100%;height:auto;display:block}
@media(max-width:700px){.columns{grid-template-columns:1fr}}
</style>
<h1>跨页表格配对复核</h1>
<p>请独立判断两张裁片是否属于同一张跨页逻辑表。缺少依据时选「无法确认」。</p>
<header><button id="prev">上一对</button><button id="next">下一对</button><select id="jump" aria-label="跳转"></select>
<span id="status" aria-live="polite"></span><label>审核人 <input id="reviewer"></label><button id="export">导出判断</button></header>
<p><button data-choice="same_table">同一逻辑表</button> <button data-choice="different_tables">两张独立表</button>
<button data-choice="uncertain">无法确认</button></p>
<div class="columns"><section><h2 id="label0"></h2><img id="image0" alt="前页表格"></section>
<section><h2 id="label1"></h2><img id="image1" alt="后页表格"></section></div>
<script id="data" type="application/json">PAYLOAD</script><script>
const x=JSON.parse(document.getElementById('data').textContent), el=id=>document.getElementById(id);
const key='tta-link-review-'+x.study_sha256;let choices={};try{choices=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
let i=0;const pairs=x.pairs;
const title={same_table:'同一逻辑表',different_tables:'两张独立表',uncertain:'无法确认'};
pairs.forEach((_,j)=>el('jump').add(new Option('第 '+(j+1)+' 对',j)));
function show(j){i=Math.max(0,Math.min(pairs.length-1,j));const p=pairs[i];el('jump').value=i;
el('status').textContent=(i+1)+' / '+pairs.length+' · 已确认 '+Object.keys(choices).length+' 对 · '+(title[choices[p.pair_id]]||'待判断');
p.tables.forEach((id,k)=>{const t=x.tables[id];el('label'+k).textContent='第 '+t.page+' 页';el('image'+k).src='data:image/webp;base64,'+t.preview;});}
el('prev').onclick=()=>show(i-1);el('next').onclick=()=>show(i+1);el('jump').onchange=()=>show(+el('jump').value);
document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{choices[pairs[i].pair_id]=b.dataset.choice;
try{localStorage.setItem(key,JSON.stringify(choices))}catch{}show(i+1);});
el('export').onclick=()=>{const reviewer=el('reviewer').value.trim();
if(!reviewer||pairs.some(p=>!choices[p.pair_id])){alert('请填审核人，并对每一对选择判断或无法确认。');return;}
const data={study_sha256:x.study_sha256,reviewer,reviewer_type:'human',reviewed_at:new Date().toISOString(),decisions:choices};
const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
const a=document.createElement('a');a.href=url;a.download='cross_page_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
if(pairs.length)show(0);else el('status').textContent='测试集未产生相邻页表格候选对。';
</script></html>"""


def prepare(dataset, output, limit):
    rows = read_split(dataset, "test")
    tables = {r["table_id"]: evidence(r) for r in rows}
    eligible, selected = select_pairs(tables, limit)
    members = {tid for _, pair in selected for tid in pair}
    tables = {tid: tables[tid] for tid in sorted(members)}
    study = dict(
        version="adjacent_page_reference_v1",
        cohort_sha256=sha((dataset / "manifest.json").read_bytes()),
        candidate_rule="Adjacent physical pages in the same document. Target two-thirds near page edges, with backfill from the other stratum. Stable hash within each stratum.",
        eligible_pairs=eligible, threshold=THRESHOLD,
        candidate_code_sha256=sha(LINKER.read_bytes()),
        tables=tables, pairs=[dict(pair_id=key, tables=pair) for key, pair in selected],
    )
    body = jbytes(study)
    immutable(output / "cross_page_study.json", body)
    gallery_tables = {}
    for tid, row in tables.items():
        image = Path(row["image"])
        if sha(image.read_bytes()) != row["image_sha256"]:
            raise ValueError(f"Changed review image {tid}")
        gallery_tables[tid] = dict(page=row["page_number"], preview=preview(image))
    payload = json.dumps(
        dict(study_sha256=sha(body), pairs=study["pairs"], tables=gallery_tables),
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    immutable(output / "cross_page_review.html", GALLERY.replace("PAYLOAD", payload).encode())
    print("cross-page review", len(selected), "of", eligible, "eligible pairs", flush=True)


def review(path, study_path):
    body = study_path.read_bytes()
    study = json.loads(body)
    record = load(path)
    if (
        record.get("study_sha256") != sha(body)
        or record.get("reviewer_type") != "human"
        or not record.get("reviewer", "").strip()
        or not record.get("reviewed_at", "").strip()
    ):
        raise ValueError("Independent human review required for this cross-page cohort")
    decisions = record["decisions"]
    if set(decisions) != {p["pair_id"] for p in study["pairs"]} or not set(decisions.values()) <= DECISIONS:
        raise ValueError("Invalid or incomplete cross-page decisions")
    return study, decisions


def logical_tables(links, refs):
    after, before = {}, {}
    for a, b in links:
        if a in after or b in before:
            raise ValueError("Conflicting confirmed continuations")
        after[a], before[b] = b, a
    tables = []
    for current in sorted(after.keys() - before.keys()):
        chain = []
        serial = []
        columns = set()
        row_offset = 0
        while current:
            row = refs[current]
            path = Path(row["gt"])
            body = path.read_bytes()
            if sha(body) != row["gt_sha256"]:
                raise ValueError("Changed page-local reference")
            reference = json.loads(body)
            text = reference["corrected_otsl"]
            grid = parse(text)
            chain.append(dict(
                table_id=current, reference=str(path), sha256=row["gt_sha256"],
                row_offset=row_offset, rows=grid.rows, columns=grid.columns,
            ))
            serial.append(text.strip())
            columns.add(grid.columns)
            row_offset += grid.rows
            current = after.get(current)
        merged = "\n".join(serial) if len(columns) == 1 else None
        if merged is not None and parse(merged).rows != row_offset:
            raise ValueError("Concatenated row count differs from the confirmed parts")
        tables.append(dict(logical_id=sha(jbytes(chain)), pages=chain, merged_otsl=merged))
    if sum(len(t["pages"]) - 1 for t in tables) != len(links):
        raise ValueError("Confirmed continuations do not form consecutive chains")
    return tables


def score(dataset, study_path, review_path, predictions, output):
    study, decisions = review(review_path, study_path)
    frozen = load(dataset / "manifest.json")
    if (
        study["cohort_sha256"] != frozen.get("review_manifest_sha256")
        or study["threshold"] != THRESHOLD
        or study["candidate_code_sha256"] != sha(LINKER.read_bytes())
        or frozen.get("cross_page") != dict(
            study_sha256=sha(study_path.read_bytes()), human_review_sha256=sha(review_path.read_bytes()),
        )
    ):
        raise ValueError("Cross-page review or threshold differs from the frozen protocol")
    refs = {row["table_id"]: row for row in read_split(dataset, "test")}
    if not set(study["tables"]) <= set(refs):
        raise ValueError("Cross-page study contains a non-test record")
    parsed = {}
    for tid, table in study["tables"].items():
        p = predictions / (tid + ".json")
        prediction = load(p)
        if prediction["table_id"] != tid or prediction["image_sha256"] != table["image_sha256"]:
            raise ValueError(f"Invalid prediction binding {tid}")
        try:
            parsed[tid] = parse(prediction["raw_otsl"])
        except ValueError:
            parsed[tid] = None
    uncertain = 0
    counts = defaultdict(lambda: [0, 0, 0])
    links, scored = [], []
    for row in study["pairs"]:
        a, b = row["tables"]
        if (
            study["tables"][a]["document_sha256"] != study["tables"][b]["document_sha256"]
            or study["tables"][b]["page_number"] != study["tables"][a]["page_number"] + 1
        ):
            raise ValueError("Pair is not adjacent within one document")
        decision = decisions[row["pair_id"]]
        if decision == "uncertain":
            uncertain += 1
            continue
        prediction = None
        if parsed[a] is not None and parsed[b] is not None:
            prediction = candidate(study["tables"][a], study["tables"][b], parsed[a], parsed[b])
        positive = bool(prediction and prediction["link_confidence"] >= THRESHOLD)
        truth = decision == "same_table"
        if truth:
            links.append((a, b))
        cluster = counts[study["tables"][a]["document_sha256"]]
        cluster[0] += positive and truth
        cluster[1] += positive and not truth
        cluster[2] += not positive and truth
        scored.append(dict(row, decision=decision, predicted=positive, evidence=prediction))
    tp, fp, fn = map(sum, zip(*counts.values())) if counts else (0, 0, 0)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output / "cross_page_pairs.jsonl", b"".join(jbytes(row) for row in scored))
    immutable(output / "logical_tables.json", jbytes(logical_tables(links, refs)))
    interval = None
    if len(counts) > 1 and tp + fn:
        rng = random.Random(42)
        document_counts = list(counts.values())
        draws = []
        for _ in range(3000):
            pos, excess, missed = map(sum, zip(*rng.choices(document_counts, k=len(counts))))
            if pos + missed:
                draws.append(2 * pos / (2 * pos + excess + missed))
        draws.sort()
        interval = [draws[int((len(draws) - 1) * q)] for q in (0.025, 0.975)] if draws else None
    metrics = dict(
        study_sha256=sha(study_path.read_bytes()), human_review_sha256=sha(review_path.read_bytes()),
        eligible_pairs=study["eligible_pairs"], reviewed=len(study["pairs"]),
        scored=len(scored), uncertain=uncertain, confirmed_links=tp + fn,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        link_f1=2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        no_link_f1=0 if tp + fn else None,
        documents=len(counts), document_bootstrap_ci_95=interval,
    )
    immutable(output / "cross_page_metrics.json", jbytes(metrics))
    print(metrics, flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=Path("data/expansion_v2/review_5k"))
    prep.add_argument("--limit", type=int, default=120)
    eval_cmd = commands.add_parser("score")
    eval_cmd.add_argument("--dataset", type=Path, default=Path("data/splits/ground_truth_v1"))
    eval_cmd.add_argument("--study", type=Path, required=True)
    eval_cmd.add_argument("--reviews", type=Path, required=True)
    eval_cmd.add_argument("--predictions", type=Path, default=Path("experiments/baseline/original_v1/B0"))
    for command in (prep, eval_cmd):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.limit < 1:
            parser.error("Review limit must be positive")
        prepare(args.dataset, args.output, args.limit)
    else:
        score(args.dataset, args.study, args.reviews, args.predictions, args.output)


if __name__ == "__main__":
    main()
