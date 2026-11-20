"""Write a portable, offline visual index of each adjudicated structure warning."""

import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from src.otsl.grid import parse


def build(destination):
    folder = Path("data/annotation/phase3_v1")
    entries = [
        json.loads(line)
        for line in (folder / "repair_ledger.jsonl").read_bytes().splitlines()
    ]
    rows = []
    for rec in sorted(
        entries, key=lambda r: (not r["original_structure_flag"], r["ordinal"])
    ):
        if not rec["proposal_sha256"]:
            continue
        tid = rec["table_id"]
        m = json.loads(Path("data/annotation/packets", tid + ".json").read_bytes())
        decision = json.loads(
            Path("data/tables/ground_truth", tid, "000001.json").read_bytes()
        )
        visual = json.loads(Path("data/annotation/raw", tid + ".json").read_bytes())
        original = parse(Path(m["raw_otsl"]).read_text()).html()
        result = decision.get("structure_tree", {}).get("html") or (
            original if decision["decision"] == "approve" else ""
        )
        with Image.open(m["image"]) as im:
            im.thumbnail((1250, 1800))
            buffer = BytesIO()
            im.convert("RGB").save(buffer, format="WEBP", quality=87, method=3)
        rows.append(
            dict(
                n=rec["ordinal"],
                suspect=rec["original_structure_flag"],
                status=rec["action"],
                source=visual["reasoning_summary"],
                original=original,
                corrected=result,
                reason=decision.get("reason", ""),
                page=m["page_number"],
                child=rec["nested_tables"],
                image=base64.b64encode(buffer.getvalue()).decode(),
            )
        )
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 3 逐项修复对照</title>
<style>body{margin:0;color:#182830;font:16px/1.6 system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:20px}h1{font-size:1.5em}h2{font-size:1.1em}
nav{display:flex;align-items:center;gap:.6em;flex-wrap:wrap}button,select,input{font:inherit;padding:.35em}.view{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.scroll{overflow:auto;max-height:78vh}img{max-width:100%;height:auto} td{border:1px solid #8999a1;padding:.25em;white-space:pre-line}table{border-collapse:collapse}small{color:#47616e} @media(max-width:760px){.view{display:block}.scroll{max-height:none}}</style>
<main><h1>Phase 3 逐项修复对照</h1><p>原审核标出的 186 项和另行核对的 91 项均可按编号查阅。原图、原识别及最终结构已内嵌在本文件中。</p><nav><button id="back">上一项</button><select id="item" aria-label="选择样本"></select><button id="next">下一项</button><label>跳转至编号 <input type="number" min="1" max="1000" id="num" style="width:5em"></label></nav><p id="state"></p><div class="view"><section><h2>原始裁剪</h2><div class="scroll"><img id="img" alt="招标文件原始表格裁剪"></div></section><section><h2>最终结构</h2><div id="fixed" class="scroll"></div></section></div><details><summary>展开原识别结果和预审意见</summary><div id="old"></div><p id="comment"></p></details></main>
<script type="application/json" id="payload">DATA</script><script>
const rows=JSON.parse(document.querySelector('#payload').textContent),el=s=>document.getElementById(s),sel=el('item');
const labels={source_structure:'依据 PDF 或原图重建',retain_reviewed_original:'核对后保留原结构',exclude_non_table:'排除版面误检'};
rows.forEach((r,i)=>sel.add(new Option('第 '+r.n+' 项 · '+labels[r.status],i)));
function show(i){i=Math.max(0,Math.min(rows.length-1,i));const r=rows[i];sel.value=i;el('num').value=r.n;el('state').textContent='第 '+r.n+' 项，来源页 '+r.page+'。'+labels[r.status]+'。'+(r.child?'包含 '+r.child+' 个内嵌表格。':'');el('img').src='data:image/webp;base64,'+r.image;el('old').innerHTML=r.original;el('fixed').innerHTML=r.corrected;if(!r.corrected)el('fixed').textContent=r.reason;el('comment').textContent=r.source;}
sel.onchange=()=>show(+sel.value);el('back').onclick=()=>show(+sel.value-1);el('next').onclick=()=>show(+sel.value+1);el('num').onchange=()=>{let i=rows.findIndex(r=>r.n===+el('num').value);if(i>=0)show(i);};show(0);
</script></html>"""
    destination.write_text(
        content.replace(
            ">DATA<",
            ">" + json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c") + "<",
        )
    )
    print(destination, destination.stat().st_size)


if __name__ == "__main__":
    build(Path("data/annotation/phase3_v1/repair_gallery.html"))
