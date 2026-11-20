"""Collect public procurement attachments linked by the publisher."""

import hashlib
import html
import re
import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import pymupdf
import requests
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/crawler.yaml")
cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
base = cfg["base"]
agent = "TenderTableResearch/1.0"
session = requests.Session()
session.headers["User-Agent"] = agent
db = sqlite3.connect("database/pilot.sqlite", timeout=45)
db.execute("pragma journal_mode=wal")
db.executescript("""
create table if not exists documents(sha256 text primary key, path text, pages integer, downloaded_at text);
create table if not exists sources(url text primary key, sha256 text, notice text, site text, project text, project_id text, kind text, version text, access text);
create table if not exists fetched(url text primary key, body text);
create table if not exists retry_queue(url text primary key, attempts integer, error text);
""")
rules = RobotFileParser()
last = 0


def fetch(url, pdf=False):
    global last
    cached = db.execute("select body from fetched where url=?", (url,)).fetchone()
    if cached and not pdf:
        return cached[0]
    if urlsplit(url).netloc != urlsplit(base).netloc or not rules.can_fetch(agent, url):
        return None
    if db.execute(
        "select attempts from retry_queue where url=?", (url,)
    ).fetchone() in [(3,)]:
        return None
    time.sleep(max(0, last + cfg["delay"] - time.monotonic()))
    try:
        with session.get(
            url, timeout=(12, 30), allow_redirects=False, stream=True
        ) as resp:
            resp.raise_for_status()
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            body = bytearray()
            for block in resp.iter_content(131072):
                body.extend(block)
                if len(body) > 50 * 1024**2:
                    raise ValueError("oversize")
        if pdf:
            return bytes(body)
        body = body.decode("utf-8", errors="replace")
        db.execute("insert or replace into fetched values(?,?)", (url, body))
        db.commit()
        return body
    except (requests.RequestException, ValueError) as exc:
        db.execute(
            "insert into retry_queue values(?,1,?) on conflict(url) do update set attempts=min(attempts+1,3),error=excluded.error",
            (url, str(exc)),
        )
        db.commit()
    finally:
        last = time.monotonic()


def anchors(body):
    for path, label in re.findall(
        r'<a\b[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body or "", re.I | re.S
    ):
        yield (
            html.unescape(path).strip(),
            " ".join(html.unescape(re.sub("<[^>]+>", "", label)).split()),
        )


def scan(index):
    for path, title in anchors(fetch(index)):
        if (
            not re.fullmatch(
                cfg.get("article_pattern", r"(?:[a-z0-9]{8,}|\d+)\.htm"), path
            )
            or not title
        ):
            continue
        notice = urljoin(index, path)
        body = fetch(notice)
        plain = html.unescape(re.sub("<[^>]*>", " ", body or ""))
        project_number = re.search(
            r"(?:项目编号|招标编号|采购编号)[：:\s]+([A-Za-z0-9][A-Za-z0-9()（）_.\-/－—\[\]]{3,})",
            plain,
        )
        project = re.search(r"项目名称[：:\s]+([^\n<>]{4,120}?)(?:\s{2,}|\n)", plain)
        for href, label in anchors(body):
            if not href.lower().endswith(".pdf") or re.search(
                "声明|登记表|承诺函|授权书|报名|营业执照|评审专家|小微企业|缴费|中小企业|成交公告|中标公告",
                label,
            ):
                continue
            if not re.search(
                "需求|采购文件|招标文件|磋商文件|谈判文件|比选文件|技术|清单|询价文件|遴选文件|招标书|评分|参数|竞价文件|分项报价|报价明细",
                label,
            ):
                continue
            url = urljoin(notice, href)
            if db.execute("select 1 from sources where url=?", (url,)).fetchone():
                continue
            blob = fetch(url, True)
            if not blob or not blob.startswith(b"%PDF-"):
                continue
            try:
                with pymupdf.open(stream=blob, filetype="pdf") as doc:
                    pages = len(doc)
                    if not pages or doc.needs_pass:
                        continue
            except pymupdf.FileDataError:
                continue
            db.execute("begin immediate")
            if count() >= cfg["target"]:
                db.rollback()
                return
            key = hashlib.sha256(blob).hexdigest()
            path = Path("data/raw/pdf") / (key + ".pdf")
            if not path.exists():
                tmp = path.with_suffix(".part")
                tmp.write_bytes(blob)
                tmp.replace(path)
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                "insert or ignore into documents values(?,?,?,?)",
                (key, str(path), pages, now),
            )
            db.execute(
                "insert into sources values(?,?,?,?,?,?,?,?,?)",
                (
                    url,
                    key,
                    notice,
                    cfg["site"],
                    project[1].strip() if project else title,
                    project_number[1] if project_number else None,
                    label,
                    key,
                    "采购单位官网公开附件，本地研究，未获再发布授权",
                ),
            )
            db.commit()
            print(count(), pages, label, flush=True)
            if count() >= cfg["target"]:
                return


def count():
    return db.execute("select count(*) from documents").fetchone()[0]


if __name__ == "__main__":
    if count() < cfg["target"]:
        robots = session.get(urljoin(base, "/robots.txt"), timeout=30)
        if robots.status_code not in (404, 410):
            robots.raise_for_status()
        rules.parse(robots.text.splitlines() if robots.status_code == 200 else [])
        last = time.monotonic()
        for n in range(100):
            for section in cfg["channels"]:
                if count() >= cfg["target"]:
                    break
                page = (
                    f"index{n or ''}.htm"
                    if "article_pattern" not in cfg
                    else ("index.html" if not n else f"index_{n + 1}.html")
                )
                scan(urljoin(base, f"{section}/{page}"))
            if count() >= cfg["target"]:
                break
    print(f"{count()}/{cfg['target']}", flush=True)
