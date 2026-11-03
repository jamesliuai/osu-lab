# Trace claims from this public release

`CLAIM_REGISTRY.json` separates supporting public evidence from historical source identity. Every cleared claim has `public_evidence` records with a path relative to the extracted source root, the SHA-256 of the actual exported bytes, a symbol/JSON locator, and the supported scope. A historical source hash is never presented as the hash of a modified export.

`historical_source_provenance` identifies earlier original-repository material, not public evidence locations. When original bytes are included, the registry separately names their exported path and hash. Unavailable originals are not dependencies for checking the scoped public claims and cannot be independently authenticated from a hash alone. Retained fixture paths inside the old immutable manifest are historical identifiers; its `copy` entries and exported map names resolve to the included data.

Run these standard-library checks from the extracted source root before adding generated outputs:

```sh
python tools/verify_release.py --strict
python tools/verify_claims.py
```

The second check resolves and hashes every public evidence entry, verifies nested generated fixtures, recomputes fidelity counts and coverage, checks public test-status totals and retained-code identity, compares the release inventory, and rejects checksum cycles. Engineering locators identify the relevant implemented functions and constants for source inspection; this does not claim GPU execution or certify all behavioral properties.

Historical validation is supported by a new explicitly derived publisher per-test summary, the unchanged prior public verification record, byte-identical CPU stdout, and the executable/test hashes. These are records of candidate_v1 execution, not a new runtime run on this revision or an independent attestation. The reader can inspect them and rerun the supplied tests. No missing native per-step events are reconstructed.

Checksums form a directed chain: package checksums hash the public manifest and payload; the manifest hashes the registry and evidence; the registry hashes evidence; evidence hashes leaf code/data. The release-scope inventory does not hash itself, the registry, manifest, or package checksums. Its path list records membership without creating a checksum dependency. External download checksums cover archives and download metadata.

Website copies use the same source-root-relative evidence paths. They resolve in the companion source ZIP, not relative to the website handoff directory. The accepted study claims resolve to unchanged public aggregate records. STUDY_PROVENANCE.json separates these public paths from omitted originals and records the existing review acceptance. The public checker uses the actual F1 structured fields, not stale planned-design labels. No additional study deliverable was requested.
