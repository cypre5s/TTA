"""Build an offline review gallery for the machine-adjudicated 5k test references."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from scripts.ground_truth import sha
from src.otsl.grid import parse


def load(path):
    return json.loads(path.read_bytes())


def preview(path):
    with Image.open(path) as image:
        image.thumbnail((1800, 1800))
        output = BytesIO()
        image.convert("RGB").save(output, format="WEBP", quality=82, method=6)
    return base64.b64encode(output.getvalue()).decode()


def render_otsl(text):
    try:
        return parse(text).html()
    except ValueError:
        from html import escape

        return "<pre>" + escape(text) + "</pre>"


def build(dataset, destination):
    manifest_path = dataset / "manifest.json"
    manifest = load(manifest_path)
    body = (dataset / "test.jsonl").read_bytes()
    if sha(body) != manifest["sha256"]["test"]:
        raise ValueError("Test review manifest changed")
    records = []
    for line in body.splitlines():
        row = json.loads(line)
        if row["tier"] == "human":
            continue
        reference_path = Path(row["reference"])
        reference = load(reference_path)
        packet = load(Path(row["packet"]))
        if (
            sha(reference_path.read_bytes()) != row["reference_sha256"]
            or sha(Path(row["packet"]).read_bytes()) != row["packet_sha256"]
            or sha(Path(row["image"]).read_bytes()) != row["image_sha256"]
        ):
            raise ValueError("Review evidence changed")
        corrected = reference["corrected_otsl"]
        original = load(Path(packet["prediction"]))["raw_otsl"]
        records.append(
            dict(
                ordinal=reference["ordinal"],
                table_id=row["table_id"],
                page=reference["page_number"],
                image=preview(Path(row["image"])),
                original=render_otsl(original),
                corrected=parse(corrected).html(),
                changed=original != corrected,
                quality=reference["visual_quality"],
                flags=reference["flags"],
                action=reference["action"],
                image_sha256=row["image_sha256"],
                reference_sha256=row["reference_sha256"],
            )
        )
    records.sort(key=lambda row: row["ordinal"])
    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    source_hash = sha(manifest_path.read_bytes())
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>5k 测试集逐项复核</title>
<style>
body{font:15px system-ui,sans-serif;color:#182532;margin:auto;max-width:1560px;padding:18px}
header,nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select,input{font:inherit;padding:5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{border:1px solid #c4ccd4;padding:10px;overflow:auto}
img{max-width:100%;max-height:60vh;object-fit:contain;object-position:top left}table{border-collapse:collapse;min-width:100%}
td{border:1px solid #687685;padding:4px;vertical-align:top}.meta{overflow-wrap:anywhere;line-height:1.55}.changed{color:#9b3d16}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style><header><h1>5k 测试集逐项复核</h1><button id="prev">上一项</button><button id="next">下一项</button>
<label>跳转 <select id="pick"></select></label><label><input id="seen" type="checkbox">已查看</label><span id="pos"></span></header>
<p id="scope"></p><p class="meta" id="meta"></p><div class="grid"><section class="panel"><h2>原始裁片</h2><img id="image"></section>
<section class="panel"><h2>最终结构</h2><div id="corrected"></div></section><section class="panel"><h2>原始 OCR 结构</h2><div id="original"></div></section>
<section class="panel"><h2>审核记录</h2><pre id="flags"></pre></section></div>
<script id="records" type="application/json">PAYLOAD</script><script>
const rows=JSON.parse(document.querySelector('#records').textContent), el=id=>document.getElementById(id), pick=el('pick');
const key='tender-5k-reviewed-SOURCE_HASH', viewed=new Set(JSON.parse(localStorage.getItem(key)||'[]'));let i=0;
rows.forEach((r,n)=>pick.add(new Option('第 '+r.ordinal+' 项',n)));
function show(n){i=Math.max(0,Math.min(rows.length-1,n));const r=rows[i];pick.value=i;el('pos').textContent=(i+1)+' / '+rows.length;
el('image').src='data:image/webp;base64,'+r.image;el('corrected').innerHTML=r.corrected;el('original').innerHTML=r.original;
el('flags').textContent=JSON.stringify(r.flags,null,2);el('meta').innerHTML='编号 '+r.ordinal+' · 页 '+r.page+' · '+r.table_id+
  ' · 质量 '+r.quality+' · 处理 '+r.action+(r.changed?' · <span class="changed">结构或文字已修正</span>':' · 保留原识别')+
  '<br>图像 '+r.image_sha256+'<br>参考 '+r.reference_sha256;el('seen').checked=viewed.has(r.table_id)}
el('prev').onclick=()=>show(i-1);el('next').onclick=()=>show(i+1);pick.onchange=()=>show(+pick.value);
el('seen').onchange=()=>{const id=rows[i].table_id;if(el('seen').checked)viewed.add(id);else viewed.delete(id);localStorage.setItem(key,JSON.stringify([...viewed]));};
el('scope').textContent='只包含新扩量中进入测试集的机器辅助参考，共 '+rows.length+' 项。清单 SHA256 SOURCE_HASH。';show(0);
</script></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        html.replace("PAYLOAD", payload).replace("SOURCE_HASH", source_hash),
        encoding="utf-8",
    )
    print(destination, len(records), destination.stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/expansion_v2/review_5k")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/expansion_v2/review_5k/test_review.html"),
    )
    arguments = parser.parse_args()
    build(arguments.dataset, arguments.output)
