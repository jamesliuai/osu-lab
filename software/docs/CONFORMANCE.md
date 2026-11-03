# Generated-content conformance

The immutable fixture manifest `tests/data/MANIFEST.json` records hashes and relative provenance for 10 generated maps and 72 native case records. Every selected map and JSON record was checked individually. No community map or old sealed test set was inspected for this export.

The historical generated-only fidelity audit reports 77,323 native steps, zero recorded mismatched steps, and equal totals for all 72 cases. This describes the recorded historical comparison against one pinned osu!lazer build. Native per-step events were not retained, so this package cannot independently replay that per-step comparison.

The original 107-test suite checks generation, converted event counts, re-simulated per-type judgement totals, max combo, step count, fixture hashes, and installed module origin. Two location tests require `OSU_SIM_EXPECT_UNDER`; without it they skip. Additional release-interface checks are reported separately in the verification report.

The fixture coverage includes normal rate and native Double Time, general maps and spinner-heavy maps, scripted perfect/sloppy/late actions. It contains only 9 Meh judgements and no slider-tick, slider-tail, or spinner misses. It does not certify spinner grades, score, positions, stacking, rendering, arbitrary maps, or universal engine equivalence. New native traces would require an instrumented native host, which is not distributed here.

The frozen fixture manifest and records are copied byte-for-byte. Source provenance paths are historical identifiers, not dependencies to discover or read during tests.
