"""Self-contained, offline index of real review packets, with links to source evidence."""

import json
from pathlib import Path


def build(path):
    rows = []
    for p in Path("data/annotation/raw").glob("*.json"):
        r = json.loads(p.read_text())
        packet = Path("data/annotation/packets") / p.name
        a = json.loads(packet.read_text())
        info = dict(
            n=a["ordinal"],
            id=r["table_id"],
            page=a["page_number"],
            summary=r["reasoning_summary"],
            valid=r["structure_valid"],
            quality=r["visual_quality"],
            image=a["image"],
            html=a["html"],
            otsl=a["raw_otsl"],
            review=str(p),
        )
        rows.append(info)
    rows.sort(key=lambda x: (x["valid"], x["n"]))
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>表格预标注审核</title>
<style>body{font:16px system-ui,sans-serif;color:#1d2935;margin:auto;max-width:1500px;padding:20px}header{display:flex;flex-wrap:wrap;gap:16px;align-items:center}h1{font-size:24px}select,button{font:inherit;padding:5px}main{display:grid;grid-template-columns:1fr 1fr;gap:14px}h2{font-size:18px}img{max-width:100%;max-height:66vh;object-fit:contain;object-position:top left}iframe{width:100%;height:65vh;border:1px solid #aab3bb}nav{display:flex;gap:16px;align-items:center;flex-wrap:wrap}p{line-height:1.6;overflow-wrap:anywhere}@media(max-width:800px){main{grid-template-columns:1fr}}</style>
<header><h1>表格预标注审核</h1><label>筛选 <select id="filter"><option value="all">全部</option><option value="bad">模型标记结构错误</option></select></label><button id="prev">上一张</button><button id="next">下一张</button><span id="pos"></span></header><p id="meta"></p>
<main><section><h2>原始裁剪</h2><a id="full"><img id="crop" alt="原始表格裁剪"></a></section><section><h2>OCR 结构预览</h2><iframe id="table" sandbox title="OTSL 转换结果"></iframe></section></main><h2>子代理审核意见</h2><p id="summary"></p><nav><a id="raw">原始 OTSL</a><a id="answer">审核 JSON</a><a id="layout">整页版面预览</a></nav>
<script type="application/json" id="records">"""
        + json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
        + """</script>
<script>const all=JSON.parse(document.getElementById('records').textContent),el=id=>document.getElementById(id);let data=all,i=0;
function show(){const r=data[i];el('pos').textContent=data.length?(i+1)+' / '+data.length:'0 / 0';if(!r){el('meta').textContent='当前筛选无样本';return}el('crop').src=el('full').href='../'+r.image;el('table').src='../'+r.html;el('raw').href='../'+r.otsl;el('answer').href='../'+r.review;el('layout').href='../data/layout/previews/'+r.id.split('-r')[0]+'.jpg';el('meta').textContent='审核序号 '+r.n+' · 文档第 '+r.page+' 页 · 图像质量 '+r.quality+' · 待人工审核';el('summary').textContent=r.summary}
el('filter').onchange=()=>{data=el('filter').value==='bad'?all.filter(r=>!r.valid):all;i=0;show()};el('prev').onclick=()=>{i=Math.max(0,i-1);show()};el('next').onclick=()=>{i=Math.min(data.length-1,i+1);show()};show();</script></html>"""
    )
    tmp = path.with_suffix(".part")
    tmp.write_text(page)
    tmp.replace(path)


if __name__ == "__main__":
    build(Path("reports/table_preview.html"))
