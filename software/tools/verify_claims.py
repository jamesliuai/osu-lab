"""Verify claims using only an extracted public source release (standard library only).

This checks declared file identities, public evidence links, finite fixture aggregates,
reported test outcomes, source interfaces, and the absence of checksum cycles. A
publisher test record is historical execution evidence, not an independent attestation
or a new test run. Historical source provenance is labelled external and never opened.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


class PublicReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(self.root.is_dir(), "release root is not a directory")
        self.graph = {}
        self.checked_hashes = 0

    def path(self, relative):
        require(isinstance(relative, str) and bool(relative), "empty/non-string public path")
        require("\\" not in relative and ":" not in relative, f"nonportable public path: {relative}")
        parsed = PurePosixPath(relative)
        require(not parsed.is_absolute() and str(parsed) == relative, f"noncanonical public path: {relative}")
        require(all(part not in (".", "..") for part in parsed.parts), f"unsafe public path: {relative}")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), f"path escapes public root: {relative}")
        require(path.is_file(), f"missing public file: {relative}")
        return path

    def read_json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def hash_reference(self, source, reference):
        require(isinstance(reference, dict), f"invalid hash reference in {source}")
        relative, expected = reference.get("path"), reference.get("sha256")
        path = self.path(relative)
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"invalid SHA-256 for {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"public evidence hash mismatch: {relative}")
        self.graph.setdefault(source, set()).add(relative)
        self.checked_hashes += 1
        return relative

    def acyclic(self):
        active, complete = [], set()

        def visit(node):
            if node in complete:
                return
            require(node not in active, "checksum cycle: " + " -> ".join(active + [node]))
            active.append(node)
            for target in sorted(self.graph.get(node, ())):
                visit(target)
            active.pop()
            complete.add(node)

        for node in sorted(self.graph):
            visit(node)
        return len(complete)


def unique_by(items, key, label):
    require(isinstance(items, list), f"{label} must be a list")
    out = {}
    for item in items:
        require(isinstance(item, dict) and key in item, f"invalid entry in {label}")
        value = item[key]
        require(isinstance(value, str) and value not in out, f"duplicate/invalid {key} in {label}: {value}")
        out[value] = item
    return out


def verify_manifest(reader):
    name = "PUBLIC_ARTIFACT_MANIFEST.json"
    manifest = reader.read_json(name)
    files = unique_by(manifest.get("files"), "path", "public manifest files")
    for path, entry in files.items():
        reader.hash_reference(name, entry)
        require(reader.path(path).stat().st_size == entry.get("bytes"), f"manifest length mismatch: {path}")
    expected = set(files) | {name, "SHA256SUMS"}
    actual = set()
    for path in reader.root.rglob("*"):
        require(path.resolve().is_relative_to(reader.root), "filesystem link escapes public root")
        if path.is_file():
            actual.add(path.relative_to(reader.root).as_posix())
    require(actual == expected, f"unlisted/missing public files: {sorted(actual ^ expected)}")
    sums = {}
    for line in reader.path("SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2, "malformed SHA256SUMS line")
        digest, path = parts
        require(path not in sums, f"duplicate checksum path: {path}")
        sums[path] = digest
        reader.hash_reference("SHA256SUMS", {"path": path, "sha256": digest})
    require(set(sums) == set(files) | {name}, "SHA256SUMS coverage differs from manifest")
    return files, actual


def verify_registry(reader):
    name = "CLAIM_REGISTRY.json"
    registry = reader.read_json(name)
    require(registry.get("schema") == "release_claim_registry_v2", "unsupported claim registry schema")
    claims = unique_by(registry.get("claims"), "id", "claims")
    for claim_id, claim in claims.items():
        require("evidence" not in claim and "evidence_sha256" not in claim,
                f"ambiguous legacy evidence fields remain in {claim_id}")
        historical = claim.get("historical_source_provenance")
        require(isinstance(historical, list), f"historical provenance not separated in {claim_id}")
        # These external identity records are never treated as public hash references.
        evidence = claim.get("public_evidence")
        require(isinstance(evidence, list), f"public evidence must be an array: {claim_id}")
        if claim.get("text") is None:
            require(str(claim.get("status", "")).startswith("pending"), f"null nonpending claim: {claim_id}")
            require(claim.get("value") is None and not evidence, f"pending claim has result/evidence: {claim_id}")
            continue
        require(bool(evidence), f"no resolvable public evidence: {claim_id}")
        seen = set()
        for ref in evidence:
            path = reader.hash_reference(name, ref)
            require(path not in seen, f"duplicate public evidence path in {claim_id}: {path}")
            seen.add(path)
            require(bool(ref.get("locator")) and bool(ref.get("supports")),
                    f"missing evidence locator/support scope: {claim_id}: {path}")
    for required in ("ENG-001", "ENG-002", "ENG-003", "FID-001", "FID-002", "REL-001", "DATA-001", "VERIFY-001"):
        require(required in claims, f"missing required claim: {required}")
    return claims


def verify_fixtures(reader, claims):
    name = "tests/data/MANIFEST.json"
    manifest = reader.read_json(name)
    maps = manifest.get("maps")
    require(isinstance(maps, dict), "fixture maps must be an object")
    map_paths = set()
    for key, entry in maps.items():
        require(re.fullmatch(r"(?:gen|spin)\d+", key) is not None, f"unexpected generated map name: {key}")
        path = f"tests/data/maps/{key}.osu"
        reader.hash_reference(name, {"path": path, "sha256": entry["sha256"]})
        map_paths.add(path)
    entries = unique_by(manifest.get("cases"), "copy", "fixture cases")
    total_steps = mismatched = 0
    totals = Counter()
    for relative, entry in entries.items():
        require(entry.get("map") in maps, f"case references an unlisted map: {relative}")
        path = "tests/data/" + relative
        reader.hash_reference(name, {"path": path, "sha256": entry["sha256"]})
        record = reader.read_json(path)
        require(record.get("totals_equal") is True, f"unequal native/simulator totals: {relative}")
        require(record.get("real_totals") == record.get("sim_totals"), f"native/simulator total mismatch: {relative}")
        require(record.get("steps_real") == record.get("steps_sim"), f"native/simulator step mismatch: {relative}")
        require(record.get("real_max_combo") == record.get("sim_max_combo"), f"native/simulator combo mismatch: {relative}")
        require(record.get("first_mismatches") == [], f"recorded first mismatches are nonempty: {relative}")
        require(isinstance(record.get("steps_real"), int) and record["steps_real"] >= 0, "invalid step count")
        require(isinstance(record.get("mismatched_steps"), int) and record["mismatched_steps"] >= 0, "invalid mismatch count")
        require(all(isinstance(v, int) and v >= 0 for v in record["real_totals"].values()), "invalid judgement count")
        total_steps += record["steps_real"]
        mismatched += record["mismatched_steps"]
        totals.update(record["real_totals"])
    derived = {"traces": len(entries), "maps": len(maps), "steps": total_steps, "recorded_mismatched_steps": mismatched}
    require(derived == claims["FID-001"]["value"], "FID-001 does not match retained fixture aggregates")
    require(derived == {"traces": 72, "maps": 10, "steps": 77323, "recorded_mismatched_steps": 0},
            "unexpected retained fixture suite")
    coverage = {"Meh": totals["Meh"], "tick_miss": totals["LargeTickMiss"],
                "tail_miss": totals["IgnoreMiss"], "spinner_miss": totals["spin_miss"]}
    require(coverage == claims["FID-002"]["value"], "FID-002 does not match native judgement categories")
    require(coverage == {"Meh": 9, "tick_miss": 0, "tail_miss": 0, "spinner_miss": 0}, "unexpected coverage totals")
    summary = reader.read_json("docs/generated_fixture_summary.json")
    require(summary["maps"] == len(maps) and summary["cases"] == len(entries), "fixture summary counts differ")
    require(summary["recorded_native_steps"] == total_steps and summary["recorded_mismatched_steps"] == mismatched,
            "fixture summary steps differ")
    require(summary["native_judgement_totals"] == dict(totals), "fixture summary judgements differ")
    return derived, map_paths


def verify_scope(reader, actual, fixture_map_paths):
    name = "evidence/release_scope_inventory.json"
    inventory = reader.read_json(name)
    paths = inventory.get("paths")
    require(isinstance(paths, list) and len(set(paths)) == len(paths), "duplicate/invalid scope path list")
    require(set(paths) | {"PUBLIC_ARTIFACT_MANIFEST.json", "SHA256SUMS"} == actual,
            "release scope inventory does not describe actual package")
    for path in paths:
        reader.path(path)
        require(not any(part in {".git", "__pycache__", ".venv", "datasets"} for part in PurePosixPath(path).parts),
                f"excluded tree present in release: {path}")
    suffixes = inventory.get("excluded_artifact_suffixes")
    require(isinstance(suffixes, list) and bool(suffixes), "missing explicit excluded artifact suffixes")
    for path in paths:
        require(PurePosixPath(path).suffix.lower() not in suffixes or path in inventory.get("allowed_document_image_paths", []), f"excluded artifact present: {path}")
    code_data = unique_by(inventory.get("code_and_data"), "path", "scope code/data")
    for path, entry in code_data.items():
        require(path not in {name, "CLAIM_REGISTRY.json", "PUBLIC_ARTIFACT_MANIFEST.json", "SHA256SUMS", "tools/verify_claims.py"},
                f"scope hash would create a metadata/self dependency: {path}")
        require(bool(entry.get("role")), f"missing code/data role: {path}")
        reader.hash_reference(name, entry)
    expected = {p for p in paths if PurePosixPath(p).suffix in {".py", ".osu"}
                or (p.startswith("tests/data/") and p.endswith(".json"))
                or (p.startswith("examples/recipe/") and p.endswith(".json"))}
    expected.discard("tools/verify_claims.py")
    require(set(code_data) == expected, f"scope code/data coverage differs: {sorted(set(code_data) ^ expected)}")
    maps = unique_by(inventory.get("map_provenance"), "path", "map provenance")
    all_maps = {p for p in paths if p.endswith(".osu")}
    require(set(maps) == all_maps and len(maps) == 13, "map provenance must cover every generated example/fixture")
    require(fixture_map_paths <= set(maps), "fixture map provenance missing")
    for path, entry in maps.items():
        reader.hash_reference(name, entry)
        require(isinstance(entry.get("recipe"), str) and bool(entry["recipe"].strip()), f"missing map generation recipe: {path}")
    return len(paths), len(code_data), len(maps)


def verify_reported_checks(reader, claims):
    name = "evidence/test_result_summary.json"
    report = reader.read_json(name)
    require(report.get("schema") == "publisher_test_record_v1", "unsupported publisher test record")
    require(report.get("runtime_tests_rerun_for_candidate_v2") is False,
            "historical execution record must not imply candidate_v2 runtime tests were rerun")
    groups = unique_by(report.get("groups"), "id", "test groups")
    expected = {"generated_conformance": 107, "release_interfaces": 7, "synthetic_analysis": 9}
    require(set(groups) == set(expected), "unexpected historical test groups")
    counts = {}
    for group_id, group in groups.items():
        cases = unique_by(group.get("cases"), "name", f"test cases {group_id}")
        require(all(c.get("status") == "passed" for c in cases.values()), f"historical nonpass status in {group_id}")
        counts[group_id] = len(cases)
    require(counts == expected, "historical test case counts differ from reported baseline")
    require(claims["VERIFY-001"]["value"] == dict(counts, skipped=0), "VERIFY-001 differs from public test record")
    cpu = report.get("cpu_example", {})
    require(cpu.get("exit_code") == 0, "CPU example did not report a successful exit")
    reader.hash_reference(name, cpu.get("log"))
    inputs = unique_by(report.get("public_inputs"), "path", "tested public inputs")
    require(bool(inputs), "test record lacks tested input identities")
    for entry in inputs.values():
        reader.hash_reference(name, entry)
    verification = reader.read_json("VERIFICATION.json")
    require(verification.get("schema") == "release_verification_v2", "unsupported release verification schema")
    runtime = verification.get("runtime_validation", {})
    require(runtime.get("runtime_tests_rerun") is False,
            "release verification must distinguish inherited runtime validation")
    runtime_refs = unique_by(runtime.get("public_evidence"), "path", "runtime evidence")
    require(set(runtime_refs) == {"evidence/candidate_v1_verification.json", name},
            "runtime evidence must include historical verification and public test summary")
    for entry in runtime_refs.values():
        reader.hash_reference("VERIFICATION.json", entry)
    baseline = reader.read_json("evidence/candidate_v1_verification.json")
    require(runtime.get("executed_candidate") == baseline.get("candidate"), "executed candidate identity differs")
    require(report.get("historically_tested_wheel", {}).get("sha256") == baseline.get("tested_wheel_sha256"),
            "historically tested wheel identity differs")
    for key, count in counts.items():
        row = runtime["test_counts"][key]
        require(row["passed"] == count and row["failed"] == 0 and row["skipped"] == 0,
                f"VERIFICATION.json differs for {key}")
        require(row == baseline["tests"][key], f"historical verification differs for {key}")
    return counts


def verify_license_scope(reader, actual):
    name = "evidence/FIXTURE_LICENSE_SCOPE.json"
    proposal = reader.read_json(name)
    require(proposal.get("schema") == "fixture_license_scope_v2", "unsupported applied fixture license scope")
    require(proposal.get("status") == "owner_approved_applied", "fixture grant must be applied")
    entries = unique_by(proposal.get("files"), "path", "fixture license proposal files")
    expected = {p for p in actual if p.startswith("tests/data/") or p.startswith("examples/maps/")}
    require(set(entries) == expected and len(entries) == 86, "fixture proposal does not match exact data inventory")
    categories = Counter()
    for path, entry in entries.items():
        reader.hash_reference(name, entry)
        require(reader.path(path).stat().st_size == entry.get("bytes"), f"proposal length mismatch: {path}")
        require(entry.get("decision_status") == "owner_approved", f"owner grant not applied: {path}")
        require(entry.get("preserve_bytes") is True, f"fixture byte-preservation condition missing: {path}")
        categories[entry.get("category")] += 1
        refs = unique_by(entry.get("public_evidence"), "path", f"fixture provenance evidence {path}")
        require(bool(refs), f"fixture proposal lacks public provenance: {path}")
        for ref in refs.values():
            reader.hash_reference(name, ref)
    require(dict(categories) == proposal.get("counts"), "fixture proposal category counts differ")
    return len(entries)


def verify_source_interfaces(reader, claims):
    symbols = {
        "src/osu_sim/generator.py": {"generate", "sample_params"},
        "src/osu_sim/rules.py": {"step_all"},
        "src/osu_sim/render.py": {"Renderer", "_launch"},
        "src/osu_sim/policy.py": {"Agent", "RecurrentCore"},
        "src/osu_sim/ppo.py": {"PPOTrainer"},
        "src/osu_sim/evaluate.py": {"SimEvaluator"},
    }
    parsed = {}
    for path, required in symbols.items():
        tree = ast.parse(reader.path(path).read_text(encoding="utf-8"))
        parsed[path] = tree
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        require(required <= names, f"engineering symbols missing: {path}: {sorted(required - names)}")
    env = ast.parse(reader.path("src/osu_sim/env.py").read_text(encoding="utf-8"))
    cls = next(n for n in env.body if isinstance(n, ast.ClassDef) and n.name == "SimEnv")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    keyword_defaults = {a.arg: v for a, v in zip(init.args.kwonlyargs, init.args.kw_defaults)}
    require(ast.literal_eval(keyword_defaults["control_hz"]) == claims["ENG-002"]["value"] == 60,
            "ENG-002 control frequency differs from exported default")
    render = parsed["src/osu_sim/render.py"]
    dimensions = None
    for node in render.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [n.id for n in node.targets[0].elts if isinstance(n, ast.Name)]
            if names == ["H_ACT", "W_ACT"]:
                dimensions = ast.literal_eval(node.value)
    require(dimensions == (192, 256) and claims["ENG-003"]["value"] == [256, 192],
            "ENG-003 dimensions differ from exported renderer")
    recipe = reader.path("examples/recipe/build_recipe_pools.py").read_text(encoding="utf-8")
    require('0.5 + 4.5 * lv' in recipe and 'pseudo difficulty, not a native star rating' in recipe,
            "DATA-001 recipe assignment/disclosure not found")


def verify(root):
    reader = PublicReader(root)
    files, actual = verify_manifest(reader)
    claims = verify_registry(reader)
    fixtures, map_paths = verify_fixtures(reader, claims)
    path_count, code_count, map_count = verify_scope(reader, actual, map_paths)
    tests = verify_reported_checks(reader, claims)
    licensed_fixture_files = verify_license_scope(reader, actual)
    verify_source_interfaces(reader, claims)
    # Resolve only the exported sibling module, including under isolated Python.
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_study_checks", reader.path("tools/verify_study.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = module.verify(reader, claims)
    for record_name, field in [("evidence/study/STUDY_PROVENANCE.json", "public_evidence"),
                               ("evidence/study/LICENSE_PROPOSAL.json", "files")]:
        record = reader.read_json(record_name)
        for reference in record[field]:
            reader.hash_reference(record_name, reference)
    spec = importlib.util.spec_from_file_location("release_document_checks", reader.path("tools/verify_documents.py"))
    document_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(document_module)
    document_checks = document_module.verify(reader, claims)
    spec = importlib.util.spec_from_file_location("applied_license_checks", reader.path("tools/verify_license_grants.py"))
    license_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(license_module)
    license_checks = license_module.verify(reader)
    project = reader.read_json("PROJECT_METADATA.json")
    for reference in project["public_evidence"]:
        reader.hash_reference("PROJECT_METADATA.json", reference)
    require(project["public_project_name"] == "osu-lab" and project["distribution_name"] == "osu-sim"
            and project["import_name"] == "osu_sim" and project["author"] == "James Liu", "project naming differs")
    require(claims["CONTRIB-001"]["text"] == project["contribution_statement"], "approved contribution differs")
    require(project["contribution_statement"] in reader.path("docs/CONTRIBUTIONS.md").read_text(encoding="utf-8"), "approved contribution missing")
    require(claims["LICENSE-001"]["value"] == project["applied_license_counts"] == license_checks["applied_license_counts"], "license claim differs")
    nodes = reader.acyclic()
    return {"status": "PASS", "payload_files": len(files), "claims": len(claims),
            "documents": document_checks, "study": study, "public_hash_references_checked": reader.checked_hashes, "hash_graph_nodes": nodes,
            "hash_graph_acyclic": True, "scope_paths": path_count, "scope_code_and_data": code_count,
            "generated_maps": map_count, "retained_fixture_aggregates": fixtures,
            "historical_passing_test_counts": tests,
            "licensed_fixture_files": licensed_fixture_files, "applied_license_checks": license_checks,
            "historical_sources_opened": False, "training_or_runtime_tests_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except (VerificationError, KeyError, TypeError, ValueError, OSError, StopIteration) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
