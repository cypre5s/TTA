"""Serialize covered, non-overlapping grids and keep nested table identities."""

from dataclasses import asdict, dataclass, field

from src.otsl.grid import TOKENS, parse


@dataclass
class CellNode:
    cell_id: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    cell_type: int = 0
    bbox: tuple[float, ...] | None = None
    query_id: int | None = None
    ocr_fragments: list[int] = field(default_factory=list)


@dataclass
class TableGraph:
    table_id: str
    rows: int
    columns: int
    cells: list[CellNode]
    bbox: tuple[float, ...] | None = None
    parent_cell_id: str | None = None
    nesting_depth: int = 0
    children: list["TableGraph"] = field(default_factory=list)


def serialize_flat(table):
    if min(table.rows, table.columns) < 1:
        raise ValueError("Empty table grid")
    slots = [[None for _ in range(table.columns)] for _ in range(table.rows)]
    for cell in table.cells:
        if (
            min(cell.row, cell.column) < 0 or min(cell.rowspan, cell.colspan) < 1
            or cell.row + cell.rowspan > table.rows or cell.column + cell.colspan > table.columns
            or TOKENS.search(cell.text)
        ):
            raise ValueError(f"Invalid span or reserved OTSL token in cell {cell.cell_id}")
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.column, cell.column + cell.colspan):
                if slots[r][c] is not None:
                    raise ValueError("Cell spans overlap")
                if r == cell.row and c == cell.column:
                    token = "<fcel>" + cell.text if cell.text.strip() else "<ecel>"
                elif r == cell.row:
                    token = "<lcel>"
                elif c == cell.column:
                    token = "<ucel>"
                else:
                    token = "<xcel>"
                slots[r][c] = token
    if any(token is None for row in slots for token in row):
        raise ValueError("Table grid has uncovered positions")
    otsl = "\n".join("".join(row) + "<nl>" for row in slots)
    return dict(
        table_id=table.table_id, rows=table.rows, columns=table.columns,
        cells=[asdict(c) for c in table.cells], bbox=table.bbox,
        otsl=otsl, html=parse(otsl).html(), parent_cell_id=table.parent_cell_id,
        nesting_depth=table.nesting_depth,
    )


def serialize_hierarchy(table):
    seen = set()
    children, links = [], []

    def visit(node, parent=None):
        if node.table_id in seen:
            raise ValueError("Duplicate child ID or cyclic hierarchy")
        seen.add(node.table_id)
        flat = serialize_flat(node)
        if parent is not None:
            if (
                node.parent_cell_id not in {c.cell_id for c in parent.cells}
                or node.nesting_depth != parent.nesting_depth + 1
            ):
                raise ValueError("Invalid child parent or nesting depth")
            children.append(flat)
            links.append(dict(relation="PARENT_CHILD", parent_table_id=parent.table_id,
                              parent_cell_id=node.parent_cell_id, child_table_id=node.table_id))
        for child in node.children:
            visit(child, node)
        return flat

    if table.parent_cell_id is not None or table.nesting_depth != 0:
        raise ValueError("Root table must have depth zero and no parent")
    parent = visit(table)
    return dict(parent_table=parent, child_tables=children, relations=links,
                parent_cell_id=[c["parent_cell_id"] for c in children],
                nesting_depth=max([0, *(c["nesting_depth"] for c in children)]),
                nested_otsl=dict(parent=parent["otsl"], children=[
                    dict(table_id=c["table_id"], parent_cell_id=c["parent_cell_id"], otsl=c["otsl"],
                         nesting_depth=c["nesting_depth"]) for c in children
                ]))
