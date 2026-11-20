"""Parse Xiaomi table tokens without silently padding or repairing the source."""

from dataclasses import dataclass, asdict
from html import escape
import re

TOKENS = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")


@dataclass
class Cell:
    row: int
    column: int
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Grid:
    rows: int
    columns: int
    cells: list[Cell]

    def as_dict(self):
        return asdict(self)

    def html(self):
        starts = {(c.row, c.column): c for c in self.cells}
        html = ["<table>"]
        for r in range(self.rows):
            html.append("<tr>")
            for col in range(self.columns):
                c = starts.get((r, col))
                if c:
                    attrs = ""
                    if c.rowspan > 1:
                        attrs += f' rowspan="{c.rowspan}"'
                    if c.colspan > 1:
                        attrs += f' colspan="{c.colspan}"'
                    html.append(f"<td{attrs}>{escape(c.text)}</td>")
            html.append("</tr>")
        return "".join(html) + "</table>"


def parse(otsl: str) -> Grid:
    matches = list(TOKENS.finditer(otsl))
    if not matches or otsl[: matches[0].start()].strip():
        raise ValueError("non-OTSL prefix or empty input")
    grid, row, cells = [], [], []
    for i, m in enumerate(matches):
        token = m[1]
        text = otsl[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else None
        ].strip()
        if token != "fcel" and text:
            raise ValueError(f"text attached to <{token}>")
        r, c = len(grid), len(row)
        if token == "nl":
            if not row:
                raise ValueError(f"empty row {r}")
            grid.append(row)
            row = []
        elif token in ("fcel", "ecel"):
            row.append(len(cells))
            cells.append(Cell(r, c, text))
        else:
            left = row[-1] if c else None
            above = grid[-1][c] if r and c < len(grid[-1]) else None
            if token == "lcel":
                anchor = left
                if anchor is None or cells[anchor].row != r:
                    raise ValueError(f"invalid left span at {r},{c}")
            elif token == "ucel":
                anchor = above
                if anchor is None or cells[anchor].column != c:
                    raise ValueError(f"invalid upper span at {r},{c}")
            else:
                anchor = left
                if anchor is None or anchor != above:
                    raise ValueError(f"invalid interior span at {r},{c}")
            row.append(anchor)
            cell = cells[anchor]
            cell.rowspan = max(cell.rowspan, r - cell.row + 1)
            cell.colspan = max(cell.colspan, c - cell.column + 1)
    if row:
        grid.append(row)
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise ValueError("ragged grid")
    for i, cell in enumerate(cells):
        if any(
            grid[r][c] != i
            for r in range(cell.row, cell.row + cell.rowspan)
            for c in range(cell.column, cell.column + cell.colspan)
        ):
            raise ValueError(f"nonrectangular span at {cell.row},{cell.column}")
    return Grid(len(grid), len(grid[0]), cells)
