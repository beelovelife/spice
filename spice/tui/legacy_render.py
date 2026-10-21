"""Plain-text Markdown rendering for the prompt-toolkit legacy TUI."""

from __future__ import annotations

import re
from io import StringIO

from rich import box
from rich.console import Console
from rich.table import Table

from spice.cli.render import _split_table_cells


def compact_blank_lines(text: str, *, max_blank_lines: int = 1) -> str:
    lines = text.splitlines()
    compacted: list[str] = []
    blank_count = 0
    for raw_line in lines:
        line = raw_line.rstrip()
        blank = not line.strip()
        if blank:
            blank_count += 1
            if blank_count > max_blank_lines:
                continue
        else:
            blank_count = 0
        compacted.append(line)
    return "\n".join(compacted)


def render_plain_markdown(text: str) -> str:
    rendered: list[str] = []
    in_code_block = False
    prev_kind: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            prev_kind = "code"
            continue
        if in_code_block:
            rendered.append(f"    {line}" if line else "")
            prev_kind = "code"
            continue
        if not stripped:
            if prev_kind == "list":
                continue
            rendered.append("")
            prev_kind = None
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            if rendered and rendered[-1].strip():
                rendered.append("")
            rendered.append(strip_inline_markdown(heading.group(2)))
            rendered.append("")
            prev_kind = "heading"
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            rendered.append(f" • {strip_inline_markdown(bullet.group(1))}")
            prev_kind = "list"
            continue
        numbered = re.match(r"^(\d+[.)])\s+(.*)$", stripped)
        if numbered:
            rendered.append(
                f" {numbered.group(1)} {strip_inline_markdown(numbered.group(2))}"
            )
            prev_kind = "list"
            continue
        if prev_kind == "list" and rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append(strip_inline_markdown(line))
        prev_kind = "text"
    return compact_blank_lines("\n".join(rendered))


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return re.sub(r"_([^_]+)_", r"\1", text)


def render_plain_table(lines: list[str], *, width: int) -> str:
    header = _split_table_cells(lines[0])
    rows = [_split_table_cells(line) for line in lines[2:]]
    column_count = (
        max(len(header), *(len(row) for row in rows)) if rows else len(header)
    )
    header += [""] * (column_count - len(header))
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=width)
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    for column in header:
        table.add_column(strip_inline_markdown(column), overflow="fold")
    for row in rows:
        cells = row + [""] * (column_count - len(row))
        table.add_row(*(strip_inline_markdown(cell) for cell in cells))
    console.print(table)
    return output.getvalue().rstrip()
