"""Verify exported document-chart hashes and plotted values using only public files.

Usage: python tools/verify_document_figures.py [--root PACKAGE_ROOT]
No imports from the simulator or private source tree are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def resolve(value, locator):
    for key in locator.split("."):
        if isinstance(value, list):
            value = value[int(key)] if key.isdecimal() else next(row for row in value if row.get("id") == key)
        else:
            value = value[key]
    return value


def public_file(root, relative_path):
    """Resolve first so traversal and symlinks cannot escape the public root."""
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Evidence path is outside the package or not a file: " + str(relative_path))
    return path


def parse_safe_svg(raw):
    # These charts are UTF-8. Decode strictly before checking, so UTF-16/null-byte
    # encodings cannot hide declarations or active markup from the preflight.
    source = raw.decode("utf-8-sig")
    if "\x00" in source:
        raise ValueError("Null bytes are not permitted in chart SVGs")
    forbidden = r"<!\s*(?:DOCTYPE|ENTITY)\b|<\s*(?:[\w.-]+:)?(?:script|style)\b|\b(?:[\w.-]+:)?href\s*=|\bstyle\s*=|\bon[a-z]+\s*=|url\s*\("
    if re.search(forbidden, source, re.IGNORECASE):
        raise ValueError("Active content, styles, links or entity declarations are not permitted in chart SVGs")
    return ET.fromstring(source)


def verify_document_figures(package_root=None):
    root = Path(package_root if package_root is not None else Path(__file__).resolve().parents[1]).resolve(strict=True)
    audit_path = public_file(root, "evidence/documents/FIGURE_AUDIT.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    analysis_path = public_file(root, audit["analysis"]["path"])
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == audit["analysis"]["sha256"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    allowed_tags = {"svg", "title", "text", "rect", "circle", "line", "polyline", "polygon"}
    counts = {}
    stems = {"fig_pd_main", "fig_pd_curves", "fig_pd_shift"}
    assert len(audit["figures"]) == 3
    assert {figure["stem"] for figure in audit["figures"]} == stems
    file_count = 0
    for figure in audit["figures"]:
        expected_files = {"evidence/study/figures/" + figure["stem"] + suffix for suffix in (".png", ".svg")}
        assert len(figure["files"]) == 2
        assert {artifact["public_path"] for artifact in figure["files"]} == expected_files
        for artifact in figure["files"]:
            path = public_file(root, artifact["public_path"])
            assert path.stat().st_size == artifact["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
            file_count += 1
        stem = figure["stem"]
        if stem == "fig_pd_main":
            continue  # Main plot retains the existing study verifier.
        svg_path = public_file(root, "evidence/study/figures/" + stem + ".svg")
        svg = parse_safe_svg(svg_path.read_bytes())
        count = 0
        for element in svg.iter():
            assert element.tag.rsplit("}", 1)[-1] in allowed_tags
            assert all(not key.lower().startswith("on") and "href" not in key.lower() for key in element.attrib)
            if "data-v" not in element.attrib:
                continue
            expected = resolve(analysis, element.attrib["data-key"])
            if stem == "fig_pd_curves":
                expected = expected[:2]  # Training decisions and Y, not accuracy/pass-rate columns.
            elif not isinstance(expected, list):
                expected = [expected]
            actual = [float(value) for value in element.attrib["data-v"].split(",")]
            assert actual == expected, (stem, element.attrib["data-key"], actual, expected)
            count += 1
        assert count == figure["numeric_annotation_checks"]
        counts[stem] = count
        if stem == "fig_pd_shift":
            text = [element.text for element in svg.iter() if element.tag.rsplit("}", 1)[-1] == "text"]
            for key in ("S6", "S7"):
                result = resolve(analysis, "secondary." + key)
                caption = f"{result['estimate']:.3f} (95% CI {result['ci95'][0]:.3f} to {result['ci95'][1]:.3f}), Holm p = {result['p_holm']:.3g}"
                assert caption in text, (key, caption)
    assert counts == {"fig_pd_curves": 72, "fig_pd_shift": 38}
    return {"status": "PASS", "analysis_hash": "PASS", "figure_file_hashes": file_count, "additional_numeric_annotations": counts, "shift_subtitles": 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    print(json.dumps(verify_document_figures(arguments.root), indent=2))
