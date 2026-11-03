"""Check the selected Study PD claims from public aggregate evidence only.

This reads no pool, checkpoint, raw evaluation, or historical source path. It
recomputes the registered fit from retained run aggregates; it does not rerun the
study or its clean-data audit and cannot independently authenticate omitted data.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


EVIDENCE = {
    "evidence/study/PD_analysis.json":
        "104d0fb82415847907af0fbc6d3293ff234ae278aa29b1e9912622405f9ac64e",
    "evidence/study/PD_supplement.json":
        "5e9f89532df85ca1c3e3aa95e78dcac54892d6390045d7881ae9c6c5342f4c2b",
    "evidence/study/PD_DEVIATIONS_2026-11-01.md":
        "a8000ea483110c187fc52e9903ba8e166b18e1fa0f478132b99f72dfcb9f49a1",
    "evidence/study/CLEAN_DATA_AUDIT_FINAL.json":
        "dae75c100c1cdde191d75272ec463fbba6c831ea554345821237908bddd8667f",
}
FIGURE = {
    "path": "evidence/study/figures/fig_pd_main.svg",
    "sha256": "6df64d0b75eace30e13b8fabcbc543e581ef261bd12e8842893b5f7c1d2c49c4",
}
PRIMARY_FIELDS = (
    "b", "ci95", "ci90", "p_one_sided", "df", "n", "n_replicates",
    "r5_category", "delta",
)
SECONDARY_FIELDS = (
    "id", "estimate", "ci95", "p_one_sided", "p_holm", "reject_holm", "df",
)
# Quantiles of Student's t distribution: two-sided 95% intervals at df 2/5,
# and the two-sided 90% primary interval at df 5. These are distribution
# constants, not estimates fitted from these observations.
T975 = {2: 4.302652729696142, 5: 2.570581835636314}
T95_DF5 = 2.0150483733330233


def require(condition, message):
    if not condition:
        raise ValueError("Study PD: " + message)


def close(actual, expected, label, tolerance=1e-10):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual), label + " is not a finite number")
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{label} differs: {actual!r} != {expected!r}")


def t_cdf(t, df):
    """Student-t CDF for positive integer df, using an exact trig recurrence.

    Substitution x = sqrt(df) tan(theta) transforms the density integral into
    C * integral(cos(theta)**(df - 1), 0, atan(abs(t)/sqrt(df))).
    """
    require(isinstance(df, int) and df > 0, "invalid Student-t df")
    theta = math.atan(abs(t) / math.sqrt(df))
    sine, cosine = math.sin(theta), math.cos(theta)
    integrals = [theta, sine]
    for m in range(2, df):
        integrals.append(sine * cosine ** (m - 1) / m
                         + (m - 1) / m * integrals[m - 2])
    normalizer = math.gamma((df + 1) / 2) / (math.sqrt(math.pi) * math.gamma(df / 2))
    area = normalizer * integrals[df - 1]
    return 0.5 - area if t < 0 else 0.5 + area


def slope_fit(runs, outcome):
    groups = {}
    for run in runs:
        groups.setdefault(run["replicate"], []).append((run["k"], outcome(run)))
    centered = []
    for values in groups.values():
        mx = sum(x for x, _ in values) / len(values)
        my = sum(y for _, y in values) / len(values)
        centered.extend((x - mx, y - my) for x, y in values)
    sxx = sum(x * x for x, _ in centered)
    b = sum(x * y for x, y in centered) / sxx
    residual_sd = math.sqrt(sum((y - b * x) ** 2 for x, y in centered)
                            / (len(runs) - len(groups) - 1))
    se = residual_sd / math.sqrt(sxx)
    df = len(runs) - len(groups) - 1
    return {"estimate": b, "se": se, "df": df, "n": len(runs),
            "residual_sd": residual_sd, "t": b / se,
            "p_one_sided": t_cdf(b / se, df)}


def paired_fit(runs, high, low):
    values = {(r["replicate"], r["N"]): r["sets"]["id"]["Y"] for r in runs}
    replicates = sorted({r["replicate"] for r in runs})
    differences = [values[r, high] - values[r, low] for r in replicates]
    n = len(differences)
    estimate = sum(differences) / n
    se = math.sqrt(sum((d - estimate) ** 2 for d in differences) / (n - 1) / n)
    return {"estimate": estimate, "se": se, "df": n - 1, "n": n,
            "t": estimate / se, "p_one_sided": t_cdf(estimate / se, n - 1)}


def check_interval(observed, estimate, se, quantile, label):
    require(isinstance(observed, list) and len(observed) == 2, label + " is not an interval")
    close(observed[0], estimate - quantile * se, label + " lower")
    close(observed[1], estimate + quantile * se, label + " upper")


def verify_figure(reader, claims, analysis, runs):
    """Check the immutable chart's declarative SVG and its numeric annotations."""
    reader.hash_reference("tools/verify_study.py", FIGURE)
    raw = reader.path(FIGURE["path"]).read_bytes()
    require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(),
            "figure contains an XML entity or document type")
    root = ET.fromstring(raw)
    namespace = "{http://www.w3.org/2000/svg}"
    tags = {"svg", "title", "rect", "line", "text", "circle", "polygon", "polyline"}
    attributes = {
        "class", "cx", "cy", "data-key", "data-v", "data-y", "fill", "font-family",
        "font-size", "font-weight", "height", "points", "r", "stroke",
        "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "stroke-width",
        "text-anchor", "transform", "viewBox", "width", "x", "x1", "x2", "y", "y1", "y2",
    }
    require(root.tag == namespace + "svg", "figure is not an SVG document")
    for node in root.iter():
        require(node.tag in {namespace + tag for tag in tags}, "figure has an unapproved SVG element")
        require(set(node.attrib) <= attributes, "figure has an unapproved SVG attribute")
        for value in node.attrib.values():
            require(not any(s in value.lower() for s in ("url(", "data:", "javascript:", "http:", "https:")),
                    "figure contains an embedded asset or external reference")
        if "font-family" in node.attrib:
            require(node.attrib["font-family"] == "Helvetica, Arial, sans-serif",
                    "figure font declaration differs; no font file is distributed")

    expected_value = {
        "by_arm": {n: {outcome: analysis["descriptive"]["by_arm"][n][outcome]
                       for outcome in ("Y_id", "Y_probe")} for n in ("16", "256", "4096")},
        "untrained_Y": analysis["descriptive"]["references"]["untrained_seed0"]["id"]["Y"],
    }
    require(claims["FIG-001"]["value"] == expected_value, "FIG-001 differs from chart source values")
    expected_records = {
        "descriptive.references.untrained_seed0.id.Y": (expected_value["untrained_Y"], "ref"),
    }
    for arm, outcomes in expected_value["by_arm"].items():
        for outcome, row in outcomes.items():
            values = {str(r["replicate"]): r["sets"]["id" if outcome == "Y_id" else "probe"]["Y"]
                      for r in runs if r["N"] == int(arm)}
            require(row["values_by_replicate"] == values and row["n"] == 3,
                    "figure arm/replicate descriptives differ from retained runs")
            mean = sum(values.values()) / 3
            se = math.sqrt(sum((y - mean) ** 2 for y in values.values()) / 2 / 3)
            close(row["mean"], mean, "figure arm mean")
            check_interval(row["ci95"], mean, se, T975[2], "figure arm mean CI95")
            prefix = f"descriptive.by_arm.{arm}.{outcome}"
            expected_records[prefix + ".mean"] = (row["mean"], "mean")
            expected_records[prefix + ".ci95"] = (row["ci95"], "ci95")
            for replicate, y in values.items():
                expected_records[prefix + ".values_by_replicate." + replicate] = (y, "rep")
    require(len(expected_records) == 31, "unexpected figure numeric record count")
    seen = set()
    subtitles = []
    for node in root.iter():
        key, text = node.get("data-key"), node.get("data-v")
        if text is None:
            if key is not None:
                require(key == "primary" and node.tag == namespace + "text"
                        and node.get("class") == "subtitle", "unknown figure nonnumeric annotation")
                subtitles.append("".join(node.itertext()))
            continue
        require(key in expected_records and key not in seen, "unknown or duplicate figure numeric key")
        seen.add(key)
        expected, kind = expected_records[key]
        require(node.get("class") == kind, "figure numeric role differs from source key")
        parts = [float(v) for v in text.split(",")]
        expected_parts = expected if isinstance(expected, list) else [expected]
        require(len(parts) == len(expected_parts), "figure numeric shape differs")
        for observed, target in zip(parts, expected_parts):
            close(observed, target, "figure " + key)
        if kind in ("ci95", "ref"):
            require(node.tag == namespace + "line", "figure interval/reference is not a line")
        else:
            expected_tag = "circle" if ".Y_id." in key else "polygon"
            require(node.tag == namespace + expected_tag, "figure point glyph differs from its series")
    require(seen == set(expected_records), "figure omits numeric records")
    p = analysis["primary"]
    headline = (f"Slope per 8x maps: b = {p['b']:.3f} "
                f"(95% CI {p['ci95'][0]:.3f} to {p['ci95'][1]:.3f}), "
                f"one-sided p = {p['p_one_sided']:.3g}; {p['r5_category']}")
    require(subtitles == [headline], "figure headline differs from primary formatting")
    return len(seen)


def verify(reader, claims):
    """Verify study evidence/claim semantics, adding leaf hashes to reader.graph."""
    for path, digest in EVIDENCE.items():
        reader.hash_reference("tools/verify_study.py", {"path": path, "sha256": digest})
    analysis = reader.read_json("evidence/study/PD_analysis.json")
    supplement = reader.read_json("evidence/study/PD_supplement.json")
    audit = reader.read_json("evidence/study/CLEAN_DATA_AUDIT_FINAL.json")
    deviations = reader.path("evidence/study/PD_DEVIATIONS_2026-11-01.md").read_text(encoding="utf-8")
    require(analysis["schema"] == "pd_analysis_result_v1", "unsupported analysis schema")
    settings = analysis["settings"]
    require(settings["arms"] == [16, 256, 4096], "actual arms are not F1")
    require(settings["replicates"] == [1, 2, 3], "actual replicates are not F1")
    require(settings["budget_decisions_D"] == 60000000, "budget differs")
    require(settings["holm_m"] == 6 and settings["secondary_family"] ==
            ["S1", "S2", "S3", "S5", "S6", "S7"], "wrong secondary family")
    require(settings["alpha"] == 0.05 and settings["delta"] == 0.15, "decision constants differ")
    require(settings["min_complete_replicates"] == 3 and
            analysis["complete_replicates"] == [1, 2, 3] and analysis["excluded_runs"] == [],
            "complete-set rule differs")
    # settings.labels is preserved historical explanatory text for the planned
    # main design. Actual settings and run rows above/below determine this F1 fit.
    require("fallback F1" in analysis["version"]["label"], "analysis version is not F1")
    runs = analysis["runs"]
    require(len(runs) == 9, "expected nine retained run aggregates")
    pairs = {(r["replicate"], r["N"]) for r in runs}
    require(pairs == {(r, n) for r in (1, 2, 3) for n in (16, 256, 4096)},
            "duplicate/missing F1 run")
    require(len({r["name"] for r in runs}) == 9, "duplicate run names")
    for run in runs:
        require(run["complete"] is True, "incomplete run in accepted fit")
        close(run["k"], math.log(run["N"] / 16, 8), "run k")
        close(settings["k"][str(run["N"])], run["k"], "settings k")
        require(set(run["sets"]) == {"id", "tempo", "pattern", "probe"}, "wrong outcome sets")
        for name, row in run["sets"].items():
            require(isinstance(row["F"], int) and row["F"] >= 0 and
                    isinstance(row["O"], int) and row["O"] > 0, "invalid aggregate F/O")
            close(row["Y"], math.log((row["F"] if row["F"] else 0.5) / row["O"]),
                  run["name"] + " " + name + " Y")
            expected_maps = {"id": 500, "tempo": 250, "pattern": 250,
                             "probe": min(run["N"], 256)}[name]
            require(row["n_maps"] == expected_maps, "wrong aggregate map count")
        close(run["G"], run["sets"]["id"]["Y"] - run["sets"]["probe"]["Y"], "G")
    primary = analysis["primary"]
    fitted = slope_fit(runs, lambda r: r["sets"]["id"]["Y"])
    require(primary["status"] == "Tested" and primary["outcome"] == "Y_id", "wrong primary outcome/status")
    for field in ("se", "df", "n", "residual_sd", "t", "p_one_sided"):
        close(primary[field], fitted[field], "primary " + field)
    close(primary["b"], fitted["estimate"], "primary b")
    require(primary["n_replicates"] == 3 and primary["delta"] == settings["delta"], "primary design differs")
    check_interval(primary["ci95"], primary["b"], fitted["se"], T975[5], "primary CI95")
    check_interval(primary["ci90"], primary["b"], fitted["se"], T95_DF5, "primary CI90")
    require(primary["p_one_sided"] < settings["alpha"] and
            primary["ci90"][1] < -settings["delta"] and
            primary["r5_category"] == "Exceeds the margin" and
            primary["practically_meaningful"] is True, "primary category is inconsistent")
    require(claims["RESULT-001"]["value"] == {k: primary[k] for k in PRIMARY_FIELDS},
            "RESULT-001 differs from primary evidence")

    secondaries = analysis["secondary"]
    require([s["id"] for s in secondaries] == settings["secondary_family"], "secondary rows differ")
    fits = {
        "S1": paired_fit(runs, 4096, 16), "S2": paired_fit(runs, 256, 16),
        "S3": paired_fit(runs, 4096, 256),
        "S5": slope_fit(runs, lambda r: r["G"]),
        "S6": slope_fit(runs, lambda r: r["sets"]["tempo"]["Y"]),
        "S7": slope_fit(runs, lambda r: r["sets"]["pattern"]["Y"]),
    }
    for secondary in secondaries:
        fit = fits[secondary["id"]]
        require(secondary["status"] == "Tested", "untested secondary")
        for field in ("estimate", "se", "df", "n", "t", "p_one_sided"):
            close(secondary[field], fit[field], secondary["id"] + " " + field)
        check_interval(secondary["ci95"], fit["estimate"], fit["se"], T975[fit["df"]],
                       secondary["id"] + " CI95")
    adjusted = 0.0
    for i, secondary in enumerate(sorted(secondaries, key=lambda s: s["p_one_sided"])):
        adjusted = max(adjusted, min(1.0, (6 - i) * secondary["p_one_sided"]))
        close(secondary["p_holm"], adjusted, secondary["id"] + " Holm p")
        require(secondary["reject_holm"] is (adjusted < settings["alpha"]), "Holm rejection differs")
    require(claims["RESULT-002"]["value"] ==
            [{k: s[k] for k in SECONDARY_FIELDS} for s in secondaries],
            "RESULT-002 differs from secondary evidence")
    expected_study = {"arms": settings["arms"], "replicates": settings["replicates"],
                      "budget_decisions": settings["budget_decisions_D"],
                      "complete_replicates": analysis["complete_replicates"], "excluded_runs": 0}
    require(claims["STUDY-001"]["value"] == expected_study, "STUDY-001 differs from actual design")
    figure_records = verify_figure(reader, claims, analysis, runs)

    require(supplement["schema"] == "pd_supplement_v1", "unsupported supplement schema")
    require(supplement["sources"]["analysis"]["sha256"] ==
            EVIDENCE["evidence/study/PD_analysis.json"], "supplement analysis identity differs")
    require(set(supplement["runs"]) == {r["name"] for r in runs}, "supplement runs differ")
    require(all(r["decisions"] >= settings["budget_decisions_D"] for r in supplement["runs"].values()),
            "supplement records a budget shortfall")
    require(audit["schema"] == "pd_clean_data_audit_v1" and audit["dry_run"] is False
            and audit["verdict"] == "PASS" and audit["failed"] == [], "audit is not a final clean-data PASS")
    audit_counts = {}
    for group, expected_count in {"a": 12, "b": 11, "c": 82, "d": 4}.items():
        rows = audit[group]
        require(len(rows) == expected_count and all(r["ok"] is True for r in rows),
                "audit group " + group + " is incomplete")
        require(audit["counts"][group] == {"rows": expected_count, "ok": expected_count},
                "audit group " + group + " count differs")
        audit_counts[group] = expected_count
    require(datetime.fromisoformat(audit["t"]) >= datetime.fromisoformat("2026-11-01T20:31:52-08:00"),
            "audit predates the accepted package")
    require("D1. Review rounds beyond three" in deviations and
            "The fifth full round (all six reviewers" in deviations and
            "is the accepted results document" in deviations, "D1 final acceptance addendum missing")
    require("I accept D1 retrospectively as a documented deviation" in deviations
            and "final internally reviewed results document" in deviations,
            "D1 owner approval missing")
    require("REVIEW-001" in claims and claims["REVIEW-001"].get("text"), "review qualification claim absent")
    return {
        "status": "PASS", "design": "F1", "aggregate_runs_checked": len(runs),
        "primary_and_secondary_fits_recomputed": 7, "holm_family_size": 6,
        "audit_record_counts": audit_counts, "immutable_original_evidence_files": len(EVIDENCE),
        "immutable_original_figures": 1, "figure_numeric_records_checked": figure_records,
        "figure_embedded_assets_or_external_references": False,
        "actual_design_fields_used": True, "historical_planned_design_labels_not_used": True,
        "historical_paths_followed": False, "raw_data_independently_authenticated": False,
        "study_training_and_clean_data_audit_rerun": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    # Only the exported sibling directory is added; this also works under -I -S.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_claims import PublicReader, verify_manifest, verify_registry
    try:
        reader = PublicReader(args.root)
        verify_manifest(reader)
        claims = verify_registry(reader)
        result = verify(reader, claims)
        result["hash_graph_nodes"] = reader.acyclic()
        result["hash_graph_acyclic"] = True
    except (ValueError, KeyError, TypeError, OSError, OverflowError, ZeroDivisionError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
