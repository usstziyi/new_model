"""把 Markdown 初稿转成 A4 双栏 Word 稿（python-docx）。

用法（在项目根目录）：
    uv run --with python-docx python plan_auto/paper/word/build_docx.py

输出：plan_auto/paper/word/manuscript_two_column.docx

版式：A4，上下 2.4 cm、左右 1.9 cm；标题/摘要通栏，正文双栏；表题在上、
图题在下；三线表用 Table Grid + 细线；正文宋体 10.5 pt，标题黑体。
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
MD = PAPER / "manuscript_计算机应用研究.md"
OUT = Path(__file__).resolve().parent / "manuscript_two_column.docx"

WIDE_FIGURES = {"cross_task.png", "permutation_null.png", "injection_curve.png"}


def latex_to_text(text: str) -> str:
    """把简单的 LaTeX 片段转成可读纯文本（用于 Word 草稿）。"""
    text = text.replace("\\mathbb{R}", "R")
    text = text.replace("\\mathrm", "")
    text = text.replace("\\mathcal{C}", "C")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "·").replace("\\times", "×")
    text = text.replace("\\dots", "…").replace("\\ldots", "…")
    text = text.replace("\\sim", "~").replace("\\approx", "≈")
    text = text.replace("\\pm", "±").replace("\\mid", "|")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\#", "#").replace("\\%", "%")
    text = text.replace("\\sum", "Σ").replace("\\in", "∈")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\tilde\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"(\1)", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", text)
    text = text.replace("_", "").replace("^", "")
    text = text.replace("$", "")
    return re.sub(r"\s+", " ", text).strip()


def display_formula_text(raw: str) -> str:
    """把论文里的 6 个展示公式映射成可读文本（Word 草稿用）。"""
    mapping = [
        ("p_i", "p_i = pos(i) / (n_r_i − 1)，p_i ∈ [0, 1]"),
        ("\\phi", "φ(p_i) = [p_i, p_i², p_i³, p_i⁴]"),
        ("\\tilde{x}", "x̃_i,c(t) = x_i,c(t) − Σ_{d=0}^{5} β_{r_i,c,t,d} p_i^d"),
        ("\\tilde{f}", "f̃_i,k = f_i,k − Σ_{d=0}^{3} γ_{r_i,k,d} p_i^d"),
        ("#\\{b", "p = (1 + #{b : M_b ≥ M_obs}) / (B + 1)"),
        ("\\mathrm{SNR}", "SNR = RMS_inj / RMS_sig"),
        ("\\text{SNR}", "SNR = RMS_inj / RMS_sig"),
    ]
    for key, text in mapping:
        if key in raw:
            return text
    return latex_to_text(raw)


def set_run(run, size=10.5, bold=False, east="宋体", color=None):
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_rich(paragraph, text, size=10.5, east="宋体", indent=True):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: latex_to_text(m.group(1)), text)
    text = text.replace("\\*", "*")
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        set_run(run, size=size, bold=(i % 2 == 1), east=east)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.15
    if indent:
        paragraph.paragraph_format.first_line_indent = Pt(size * 2)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run(run, size=12 if level == 1 else 10.5, bold=True, east="黑体")
    p.paragraph_format.space_before = Pt(6 if level == 1 else 3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    return p


def set_columns(section, num: int, space_twips: int = 400):
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    if cols:
        col = cols[0]
    else:
        col = OxmlElement("w:cols")
        sect_pr.append(col)
    col.set(qn("w:num"), str(num))
    col.set(qn("w:space"), str(space_twips))
    col.set(qn("w:equalWidth"), "1")


def new_section(doc, num_columns: int):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.4)
    set_columns(section, num_columns)
    return section


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run(run, size=9)


def add_caption(doc, text, size=9):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run(run, size=size, bold=True, east="黑体")
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run(run, size=8.5, east="宋体")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0


def add_image(doc, path: Path, caption: str, wide: bool):
    if wide:
        new_section(doc, 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(16.5) if wide else Cm(8.2))
    add_caption(doc, caption)
    if wide:
        new_section(doc, 2)


def _set_cell_borders(cell, top=None, bottom=None):
    """top/bottom: None 表示无边框，否则为线宽（1/8 pt）。"""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, width in (("top", top), ("bottom", bottom), ("left", None), ("right", None)):
        el = OxmlElement(f"w:{edge}")
        if width is None:
            el.set(qn("w:val"), "nil")
        else:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(width))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
        borders.append(el)
    tc_pr.append(borders)


def add_table(doc, caption: str, rows: list[list[str]], note: str | None):
    add_caption(doc, caption)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_width = Cm(8.0 / len(rows[0]))
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            cell.width = col_width
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or j > 0) else WD_ALIGN_PARAGRAPH.LEFT
            cell_text = re.sub(r"\*\*", "", cell_text)
            cell_text = re.sub(r"\$([^$]+)\$", lambda m: latex_to_text(m.group(1)), cell_text)
            run = p.add_run(cell_text)
            set_run(run, size=8, bold=(i == 0), east="宋体")
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            # 三线表：首行上粗线 + 表头下细线，末行下粗线，其余无竖线
            top = 14 if i == 0 else None
            bottom = 6 if i == 0 else (14 if i == len(rows) - 1 else None)
            _set_cell_borders(cell, top=top, bottom=bottom)
    if note:
        add_note(doc, note)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows, i = [], start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def main() -> None:
    lines = MD.read_text(encoding="utf-8").splitlines()
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.4)
    set_columns(section, 1)
    add_page_number(section)

    doc.core_properties.title = "面向想象语音语义解码的呈现顺序混杂诊断与校正方法"
    doc.core_properties.author = "（待填）"

    in_english_block = False
    body_started = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped == "---" or re.fullmatch(r"[*\-]{3,}", stripped):
            i += 1
            continue

        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(stripped[2:])
            set_run(run, size=16, bold=True, east="黑体")
            p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        if stripped.startswith("**作者**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_rich(p, stripped, size=10.5, indent=False)
            i += 1
            continue

        if stripped.startswith("## 摘要"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("摘　要")
            set_run(run, size=10.5, bold=True, east="黑体")
            i += 1
            continue

        if stripped.startswith("## Abstract"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("Abstract")
            set_run(run, size=10.5, bold=True, east="黑体")
            in_english_block = True
            i += 1
            continue

        if stripped.startswith("## ") and "参考文献" in stripped:
            add_heading(doc, "参考文献", 1)
            i += 1
            continue

        if stripped.startswith("## "):
            if not body_started:
                new_section(doc, 2)
                body_started = True
            add_heading(doc, stripped[3:].strip(), 1)
            i += 1
            continue

        if stripped.startswith("### "):
            add_heading(doc, stripped[4:].strip(), 2)
            i += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                caption, rel = match.group(1), match.group(2)
                image_path = (PAPER / rel).resolve()
                add_image(doc, image_path, caption, image_path.name in WIDE_FIGURES)
            i += 1
            continue

        if stripped.startswith("**表"):
            caption = stripped.strip("*").strip()
            rows, next_i = parse_table(lines, i + 2)
            note = None
            if next_i < len(lines) and lines[next_i].strip().startswith("注："):
                note = lines[next_i].strip()
                next_i += 1
            if rows:
                add_table(doc, caption, rows, note)
            i = next_i
            continue

        if stripped.startswith("$$"):
            block = [stripped[2:]]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                block.append(lines[i].strip())
                i += 1
            if i < len(lines):
                block.append(lines[i].strip().replace("$$", ""))
                i += 1
            text = display_formula_text(" ".join(block))
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            set_run(run, size=10, bold=False, east="宋体")
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            continue

        if stripped.startswith("> "):
            note = re.sub(r"[*`]", "", stripped[2:])
            add_note(doc, note)
            i += 1
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_rich(p, stripped[2:], size=10.5)
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            p = doc.add_paragraph()
            add_rich(p, stripped, size=10.5)
            i += 1
            continue

        if stripped.startswith("[") and re.match(r"^\[\d+\]", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(9)
            p.paragraph_format.first_line_indent = Pt(-9)
            add_rich(p, stripped, size=9, indent=False)
            i += 1
            continue

        # 普通段落：英文块中的标题/作者用居中或小字号
        if in_english_block and stripped.startswith("**Diagnosing"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_rich(p, stripped, size=12, indent=False)
            i += 1
            continue
        if in_english_block and stripped.startswith("**Authors"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_rich(p, stripped, size=9, indent=False)
            i += 1
            continue

        if stripped.startswith(("**关键词**", "**中图分类号**", "**基金项目**", "**Abstract**", "**Keywords**", "**Foundation item**")):
            p = doc.add_paragraph()
            add_rich(p, stripped, size=10.5, indent=False)
            i += 1
            continue

        p = doc.add_paragraph()
        add_rich(p, stripped, size=10.5, east="宋体")
        i += 1

    doc.save(OUT)
    print(f"[Docx] 写入 {OUT}")


if __name__ == "__main__":
    main()
