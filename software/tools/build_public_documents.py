"""Render the release's audited Markdown documents without a browser.

Requires reportlab, Pillow and pypdf. Run from any directory:
    python tools/build_public_documents.py --root PATH_TO_RELEASE --output-dir rebuilt_documents
The default outputs are docs/REPORT.pdf and docs/ONE_PAGE_SUMMARY.pdf. Relative
--output-dir paths resolve under --root; absolute output paths also work. Existing
PDFs are never replaced unless --overwrite is explicitly supplied. Font inputs
default to the Bitstream Vera family distributed with ReportLab. The release's
exact font notice is attached to each PDF so a standalone copy carries it.
This renders supplied content; it does not run the study or fetch dependencies.
"""
from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path
import re

from PIL import Image as PILImage
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from pypdf import PdfReader, PdfWriter


def inline(value: str) -> str:
    """Escape prose first, then interpret the small audited Markdown subset."""
    value = re.sub(r"</?div\b[^>]*>", "", value)
    value = value.replace("<strong>", "**").replace("</strong>", "**")
    value = value.replace("<em>", "*").replace("</em>", "*")
    stash: list[str] = []

    def code(match: re.Match[str]) -> str:
        stash.append('<font name="Vera">' + html.escape(match[1]) + "</font>")
        return f"ZZINLINECODE{len(stash) - 1}ZZ"

    value = re.sub(r"`([^`]+)`", code, value)
    value = html.escape(value)

    def link(match: re.Match[str]) -> str:
        label, target = match[1], html.unescape(match[2])
        if target.startswith(("https://", "http://")):
            return '<link href="' + html.escape(target, quote=True) + '" color="#17496b">' + label + "</link>"
        # Local destinations are useful in the source archive; keep them visible
        # in a standalone PDF, rather than implying a working remote download.
        return label if label == html.escape(target) else label + " (" + html.escape(target) + ")"

    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", value)
    for i, fragment in enumerate(stash):
        value = value.replace(f"ZZINLINECODE{i}ZZ", fragment)
    return value


def register_fonts(font_dir: Path) -> None:
    for name, filename in [
        ("Vera", "Vera.ttf"), ("Vera-Bold", "VeraBd.ttf"),
        ("Vera-Italic", "VeraIt.ttf"), ("Vera-BoldItalic", "VeraBI.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "Vera", normal="Vera", bold="Vera-Bold", italic="Vera-Italic",
        boldItalic="Vera-BoldItalic",
    )


def styles(summary: bool, scale: float) -> dict[str, ParagraphStyle]:
    size = (9.1 if summary else 9.8) * scale
    lead = (11.5 if summary else 14.1) * scale
    body = ParagraphStyle(
        "body", fontName="Vera", fontSize=size, leading=lead,
        textColor=colors.HexColor("#172630"), spaceAfter=5.0 * scale,
        splitLongWords=1, allowWidows=0, allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=body, fontName="Vera-Bold",
            fontSize=(18 if summary else 19.5) * scale,
            leading=(22 if summary else 25) * scale, spaceAfter=13 * scale,
            textColor=colors.HexColor("#10374b"), keepWithNext=True,
        ),
        "heading": ParagraphStyle(
            "heading", parent=body, fontName="Vera-Bold", fontSize=12.5 * scale,
            leading=16 * scale, spaceBefore=12 * scale, spaceAfter=7 * scale,
            keepWithNext=True, textColor=colors.HexColor("#10374b"),
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=body, fontName="Vera-Bold", fontSize=10.5 * scale,
            leading=14 * scale, spaceBefore=9 * scale, keepWithNext=True,
        ),
        "list": ParagraphStyle(
            "list", parent=body, leftIndent=12, firstLineIndent=-12,
        ),
        "caption": ParagraphStyle(
            "caption", parent=body, fontSize=(8.1 if summary else 8.6) * scale,
            leading=(10.2 if summary else 11.4) * scale, spaceAfter=9 * scale,
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=8 * scale, leading=10.5 * scale,
            spaceAfter=0, allowWidows=1, allowOrphans=1,
        ),
        "header_cell": ParagraphStyle(
            "header_cell", parent=body, fontName="Vera-Bold", fontSize=8 * scale,
            leading=10.5 * scale, spaceAfter=0, textColor=colors.HexColor("#10374b"),
            allowWidows=1, allowOrphans=1,
        ),
    }


def image_path(line: str) -> str | None:
    match = re.match(r'^<img\s+.*?src=["\']([^"\']+)["\']', line)
    if match:
        return html.unescape(match[1])
    match = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", line)
    return match[1] if match else None


def parse_blocks(source: str) -> list[tuple[str, object]]:
    lines = source.splitlines()
    blocks: list[tuple[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            end = i + 1
            while end < len(lines) and not lines[end].strip().startswith("```"):
                end += 1
            if end == len(lines):
                raise ValueError("Unclosed code fence")
            for code_line in lines[i + 1:end]:
                blocks.append(("code", code_line))
            i = end + 1
            continue
        if re.match(r"^#{1,6}\s", line):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(("h" + str(level), line[level:].strip()))
            i += 1
            continue
        img = image_path(line)
        if img:
            blocks.append(("image", img))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            if not rows or len({len(r) for r in rows}) != 1:
                raise ValueError("Malformed Markdown table")
            blocks.append(("table", rows))
            continue
        kind = "list" if re.match(r"^(?:[-*]|\d+\.)\s", line) else "body"
        pieces = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            nxt = lines[i].strip()
            if (re.match(r"^(?:#{1,6}\s|[-*]\s|\d+\.\s|\||```)", nxt)
                    or image_path(nxt)):
                break
            pieces.append(nxt)
            i += 1
        text = " ".join(pieces)
        if kind == "list" and text.startswith(("- ", "* ")):
            text = "• " + text[2:]
        if kind == "body" and text.startswith("*") and text.endswith("*") and not text.startswith("**"):
            kind = "caption"
        blocks.append((kind, text))
    return blocks


def table_widths(rows: list[list[str]], width: float) -> list[float]:
    n = len(rows[0])
    # Hash inventories need room for the 64-character identities. Long cells
    # are still permitted to wrap, preserving every character.
    if n == 2 and any("SHA-256" in x for x in rows[0]):
        return [width * .42, width * .58]
    if n == 3 and "Time (UTC)" in rows[0]:
        return [width * .43, width * .15, width * .42]
    if n == 2:
        return [width * .34, width * .66]
    return [width / n] * n


def flowables(blocks, root: Path, source_path: Path, width: float, summary: bool, scale: float):
    sty = styles(summary, scale)
    result = []
    images = []
    for kind, content in blocks:
        if kind.startswith("h") and kind[1:].isdigit():
            style = sty["title" if kind == "h1" else "heading" if kind == "h2" else "subheading"]
            result.append(Paragraph(inline(content), style))
        elif kind == "table":
            cells = [[Paragraph(inline(c), sty["header_cell"] if row_i == 0 else sty["cell"])
                      for c in row] for row_i, row in enumerate(content)]
            table = Table(cells, colWidths=table_widths(content, width), repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f3")),
                ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#8ba4af")),
                ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#d5dfe4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            result.extend([table, Spacer(1, 8 * scale)])
        elif kind == "image":
            path = (source_path.parent / content).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Image must resolve inside the public release: {content}")
            if path.suffix.lower() != ".png":
                raise ValueError("The audited document renderer accepts PNG figures only")
            with PILImage.open(path) as bitmap:
                iw, ih = bitmap.size
            draw_width = min(width, (335 if summary else width) * scale)
            pic = Image(str(path), width=draw_width, height=draw_width * ih / iw)
            pic.hAlign = "CENTER"
            result.extend([Spacer(1, 4 * scale), pic, Spacer(1, 5 * scale)])
            images.append(path.relative_to(root).as_posix())
        elif kind == "code":
            result.append(Paragraph(html.escape(content) or " ", sty["cell"]))
        else:
            result.append(Paragraph(inline(content), sty[kind]))
    # A figure and its immediately following caption travel together.
    grouped = []
    j = 0
    while j < len(result):
        if isinstance(result[j], Image) and j + 2 < len(result) and isinstance(result[j + 2], Paragraph):
            if result[j + 2].style.name == "caption":
                grouped.append(KeepTogether(result[j:j + 3]))
                j += 3
                continue
        grouped.append(result[j])
        j += 1
    return grouped, images


def render(root: Path, output_dir: Path, filename: str, font_notice: bytes, overwrite: bool) -> dict:
    source = root / "docs" / (filename + ".md")
    target = output_dir / (filename + ".pdf")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; use a new candidate or --overwrite")
    source_text = source.read_text(encoding="utf-8")
    author_match = re.search(r"^Author: (.+)$", source_text, re.M)
    author = author_match[1].strip() if author_match else ""
    # ReportLab otherwise substitutes a missing glyph silently. Fail clearly
    # if future editorial content needs a font outside this audited family.
    required = set(source_text) - set("\r\n\t")
    for font in ["Vera", "Vera-Bold", "Vera-Italic", "Vera-BoldItalic"]:
        missing = sorted(c for c in required if ord(c) not in pdfmetrics.getFont(font).face.charToGlyph)
        if missing:
            raise ValueError(f"Unsupported characters in {font}: {missing!r}")
    blocks = parse_blocks(source_text)
    summary = filename == "ONE_PAGE_SUMMARY"
    title = next(str(text) for kind, text in blocks if kind == "h1")
    margin = 37 if summary else 48
    width = A4[0] - 2 * margin
    final = None
    used_scale = None
    # Summary content is never dropped: lower scale within a bounded readable
    # range, and fail explicitly if it still cannot fit on one page.
    scales = [1, .98, .96, .94, .92, .90] if summary else [1]
    for scale in scales:
        stream = io.BytesIO()
        doc = SimpleDocTemplate(
            stream, pagesize=A4, leftMargin=margin, rightMargin=margin,
            topMargin=31 if summary else 42, bottomMargin=34 if summary else 40,
            title=title, author=author, creator="Public document renderer", invariant=1,
        )
        story, images = flowables(blocks, root, source, width, summary, scale)

        def footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Vera", 7)
            canvas.setFillColor(colors.HexColor("#526976"))
            canvas.drawString(margin, 19, "osu-lab · Study PD · Public document derivative")
            canvas.drawRightString(A4[0] - margin, 19, str(document.page))
            canvas.restoreState()

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        pdf = PdfReader(io.BytesIO(stream.getvalue()))
        if summary and len(pdf.pages) != 1:
            continue
        final = pdf
        used_scale = scale
        break
    if final is None:
        raise ValueError("Summary exceeds one page at the minimum readable layout; revise layout explicitly")
    writer = PdfWriter(clone_from=final)
    writer.add_metadata({"/Title": title, "/Author": author, "/Creator": "Public document renderer",
                         "/Subject": "Public derivative; provenance and limitations are supplied in the source release"})
    writer.add_attachment("Bitstream-Vera.txt", font_notice)
    for license_path in ["LICENSE-STUDY.md", "licenses/CC-BY-4.0.txt"]:
        writer.add_attachment(Path(license_path).name, (root / license_path).read_bytes())
    with target.open("wb" if overwrite else "xb") as output:
        writer.write(output)
    checked = PdfReader(target)
    assert len(checked.pages) == len(final.pages)
    assert checked.attachments["Bitstream-Vera.txt"][0] == font_notice
    output_label = target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)
    return {"source": source.relative_to(root).as_posix(), "pdf": output_label,
            "pages": len(checked.pages), "blocks": len(blocks), "images": images,
            "body_font_points": round((9.1 if summary else 9.8) * used_scale, 3),
            "font_notice_attached": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted public release root")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"),
                        help="PDF destination; relative to --root unless absolute (default: docs)")
    parser.add_argument("--font-dir", type=Path, default=Path(reportlab.__file__).parent / "fonts")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing PDFs in this root")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else root / args.output_dir).resolve()
    for name in ["REPORT", "ONE_PAGE_SUMMARY"]:
        if not (root / "docs" / (name + ".md")).is_file():
            raise FileNotFoundError(f"Missing public Markdown: {name}")
        if not args.overwrite and (output_dir / (name + ".pdf")).exists():
            raise FileExistsError(f"Refusing to overwrite {name}.pdf; use a new candidate or --overwrite")
    register_fonts(args.font_dir.resolve())
    notice = (root / "licenses" / "Bitstream-Vera.txt").read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [render(root, output_dir, name, notice, args.overwrite) for name in ["REPORT", "ONE_PAGE_SUMMARY"]]
    print(json.dumps({"status": "rendered", "documents": records}, indent=2))


if __name__ == "__main__":
    main()
