"""Shared fixtures of the osu-sim conformance tests (generated content only).

The tests import the INSTALLED osu_sim (they never put a source checkout on sys.path). Set OSU_SIM_EXPECT_UNDER to a
directory to assert that the imported package lies under it (the M5 offline install check does).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))


def pytest_report_header(config):
    import osu_sim
    lines = [f"osu_sim imported from {Path(osu_sim.__file__).resolve()}"]
    want = os.environ.get("OSU_SIM_EXPECT_UNDER")
    if want:
        lines.append(f"OSU_SIM_EXPECT_UNDER={want}")
    return lines
