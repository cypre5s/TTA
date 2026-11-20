"""Extract the complete labelled paragraph, including text across inline fonts."""

from bs4 import BeautifulSoup
import re


def notice_metadata(body, current):
    soup = BeautifulSoup(body, "html.parser")
    texts = [
        " ".join(x.stripped_strings).strip() for x in soup.find_all(["p", "h2", "h3"])
    ]
    title = re.compile(
        r"^(?:[一二三四五六七八九][、.．]\s*|\d+[、.．]\s*)?项目名称\s*[：:]\s*(.+)"
    )
    number = re.compile(
        r"^(?:[一二三四五六七八九][、.．]\s*|\d+[、.．]\s*)?(?:采购项目编号|项目编号|采购编号|招标编号)\s*[：:]\s*(.+)"
    )
    for field, pattern in [("project", title), ("project_id", number)]:
        for t in texts:
            m = pattern.match(t)
            if not m:
                continue
            v = m[1].strip()
            v = re.split(
                r"\s*(?:[（(]\s*(?:招标文件编号|项目编号|采购编号)|(?:二|三|四)、|项目名称\s*[：:]|预算金额\s*[：:])",
                v,
            )[0]
            v = v.strip(" ：:。；;\u200b")
            if 1 < len(v) < 250:
                current[field] = v
                break
    return current
