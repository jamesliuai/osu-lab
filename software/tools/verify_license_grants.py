"""Verify applied owner grants against the exact approved candidate_v4 scope.

Standard library only. Reads exported records/files, never historical source
paths. This validates scope and record consistency, not an independent legal
ownership opinion. Publication remains a separate owner instruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


LEDGER = {
    "path": "evidence/APPROVED_V4_LICENSE_SCOPE.json",
    "sha256": "9cf6ae185e9b0ad13927a5d97e052b3c5ba4c96cec68a5aa6efe60a9a8c40adf",
}
RIGHTS_BOUNDARY = {
    "rights_holder": "James Liu",
    "only_rights_held": True,
    "third_party_rights_granted": False,
    "excluded_assets_granted": False,
    "code_relicensed": False,
}
CHANGING_DOCUMENTS = {
    "docs/REPORT.md", "docs/REPORT.pdf",
    "docs/ONE_PAGE_SUMMARY.md", "docs/ONE_PAGE_SUMMARY.pdf",
}


def require(condition, message):
    if not condition:
        raise ValueError("License grants: " + message)


def keyed(rows, label):
    require(isinstance(rows, list), label + " must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("path"), str), label + " invalid row")
        require(row["path"] not in result, label + " duplicate path: " + row["path"])
        result[row["path"]] = row
    return result


def verify(reader):
    reader.hash_reference("tools/verify_license_grants.py", LEDGER)
    ledger = reader.read_json(LEDGER["path"])
    require(ledger.get("schema") == "approved_candidate_v4_license_scope_v1", "unsupported approved-scope schema")
    require(ledger.get("rights_boundary") == RIGHTS_BOUNDARY, "approved rights boundary differs")
    approved = keyed(ledger.get("approved_files"), "approved candidate_v4 files")
    require(len(approved) == 105, "approved scope is not exactly 105 files")
    expected_counts = {"CC0-1.0": 92, "CC-BY-4.0": 13}
    require(dict(Counter(row["license"] for row in approved.values())) == expected_counts,
            "approved license split differs")
    inventories = [
        ("evidence/FIXTURE_LICENSE_SCOPE.json", "fixture_license_scope_v2", "fixtures_examples", 86),
        ("evidence/study/LICENSE_PROPOSAL.json", "study_evidence_license_scope_v2", "study_documents", 19),
    ]
    checked = Counter()
    unchanged = 0
    changed_documents = []
    for name, schema, group, count in inventories:
        inventory = reader.read_json(name)
        require(inventory.get("schema") == schema, "unsupported applied inventory: " + name)
        require(inventory.get("status") == "owner_approved_applied", "grant is not applied: " + name)
        require(inventory.get("licensor") == "James Liu", "licensor differs: " + name)
        require(inventory.get("rights_boundary") == RIGHTS_BOUNDARY, "rights boundary differs: " + name)
        entries = keyed(inventory.get("files"), "applied files in " + name)
        expected = {path for path, row in approved.items() if row["scope_group"] == group}
        require(set(entries) == expected and len(entries) == count,
                "applied scope expands or differs from approved candidate_v4: " + name)
        for path, entry in entries.items():
            prior = approved[path]
            require(entry.get("decision_status") == "owner_approved", "owner decision not applied: " + path)
            require(entry.get("license") == prior["license"], "license assignment differs: " + path)
            require(entry.get("approved_candidate_v4_sha256") == prior["approved_candidate_v4_sha256"],
                    "approved candidate_v4 identity differs: " + path)
            reader.hash_reference(name, entry)
            require(reader.path(path).stat().st_size == entry.get("bytes"), "current length differs: " + path)
            identical = (entry["sha256"] == prior["approved_candidate_v4_sha256"]
                         and entry["bytes"] == prior["approved_candidate_v4_bytes"])
            if path not in CHANGING_DOCUMENTS:
                require(identical, "approved frozen/data/chart record changed: " + path)
            elif not identical:
                require(entry["license"] == "CC-BY-4.0", "edited document lacks approved source license")
                changed_documents.append(path)
            if identical:
                unchanged += 1
            checked[entry["license"]] += 1
    require(dict(checked) == expected_counts, "applied overall license split differs")
    require(unchanged >= 101, "fewer than 101 approved files remain unchanged")
    return {
        "status": "PASS",
        "approved_scope_files": 105,
        "applied_license_counts": dict(checked),
        "fixture_example_files": 86,
        "study_factual_bibliographic_CC0_files": 6,
        "study_document_chart_CC_BY_files": 13,
        "byte_identical_to_approved_candidate_v4": unchanged,
        "changed_authorized_document_derivatives": sorted(changed_documents),
        "rights_limited_to_owner_held_rights": True,
        "third_party_code_and_excluded_assets_relicensed": False,
        "historical_source_paths_opened": False,
        "publication_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_claims import PublicReader
    try:
        print(json.dumps(verify(PublicReader(arguments.root)), indent=2, sort_keys=True))
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
