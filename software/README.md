# osu-lab

A procedural-map reinforcement-learning research project by **James Liu**. The Python distribution remains **osu-sim** and the import package remains **osu_sim**.

Generate maps and run osu!standard judgement simulations on the CPU. The source also includes an approximate asset-free renderer, calibrated GPU renderer code, a recurrent policy, PPO training, and evaluation. Rules and conversion adapt upstream osu! and osu-framework logic; see [third-party notices](THIRD_PARTY_NOTICES.md).

This local candidate contains generated examples and generated conformance records. It includes no trained weights, community beatmaps, music, textures, or font atlases. The accepted procedural Study PD results are included as scoped aggregate evidence; see [results and limitations](docs/STUDY_RESULTS.md). Publication has not been authorized.

## CPU quick-start

Use Python 3.12 in a fresh environment. From this extracted source release:

```sh
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` on PowerShell, or `source .venv/bin/activate` on a POSIX shell), then:

```sh
python -m pip install ".[test]"
python examples/generated_data_example.py --out-dir demo_output
```

The example generates maps, compiles them, and plays scripted on-time and late actions on the CPU. Printed judgement counts are a software demonstration, not a learned-policy result. It needs no game client, external assets, checkpoint, or GPU. Installation fetches dependencies; the example itself needs no network. See [environment](docs/ENVIRONMENT.md) for the exact local verification environment and offline installation.

## Checks and scope

```sh
python -m pytest tests -p no:cacheprovider
```

For a strict installed-copy check, follow [reproduction](docs/REPRODUCTION.md). The retained generated conformance data covers 72 scripted cases on 10 maps. Public tests compare judgement totals, max combo, and step count. Native per-step events were not retained. No universal fidelity or human-skill claim follows; see [limitations](docs/LIMITATIONS.md).

- [Environment and interface](docs/ENVIRONMENT.md)
- [Training and evaluation](docs/TRAINING.md)
- [Reproduction and analysis](docs/REPRODUCTION.md)
- [Conformance evidence](docs/CONFORMANCE.md)
- [Assets and license scope](docs/ASSETS_AND_LICENSE_SCOPE.md)
- [Release changes](docs/RELEASE_CHANGES.md)
- [Verified claims](CLAIM_REGISTRY.json) and [artifact inventory](PUBLIC_ARTIFACT_MANIFEST.json)

The absolute cursor can cross the playfield in one step. Generator difficulty labels are synthetic sampling labels, not native star ratings. Full-renderer training has separately licensed asset dependencies that this release omits.

Public claim provenance can be checked without the private repository: `python tools/verify_claims.py`. See [how to trace claims](docs/CLAIM_PROVENANCE.md). Runtime pass counts describe candidate_v1 execution carried forward only for byte-identical retained code and tests; new metadata is verified separately.

## Study documents

Read the public [report](docs/REPORT.pdf) ([Markdown](docs/REPORT.md)) and [one-page summary](docs/ONE_PAGE_SUMMARY.pdf) ([Markdown](docs/ONE_PAGE_SUMMARY.md)). These are editorial derivatives of the current accepted handover. The owner approved D1 retrospectively; review exceeded the preregistered three-round cap and remained internal, not external peer review.

Public [document provenance](evidence/documents/DOCUMENT_PROVENANCE.json) distinguishes historical source identities from exported evidence. The report preserves confirmatory, descriptive and exploratory distinctions; omitted raw inputs and media limit independent reproduction. All public numerical claims are scoped by CLAIM_REGISTRY.json. The owner-approved licenses are applied to their exact scopes; see [data rights](LICENSE-DATA.md) and [study/document rights](LICENSE-STUDY.md). This local preparation does not authorize publication.

Run `python tools/verify_documents.py` for document-evidence and figure checks. Optional PDF rebuilding requires ReportLab and pypdf (obtained separately): `python tools/build_public_documents.py --root . --output-dir rebuilt_documents`. The renderer uses the bundled ReportLab Vera font files and includes their permission notice; it does not use the game's renderer or assets. PDF byte identity may vary with dependency versions.

## Personal contribution

I developed and integrated a reinforcement-learning system for procedurally generated osu! maps, and designed and ran a study of how practice-map diversity affects generalization. The system adapts upstream osu! and osu-framework rules and uses established PPO and recurrent-network methods.

See [contribution and upstream attribution](docs/CONTRIBUTIONS.md). Adapted upstream components retain their separate copyright/license notices.

## Licenses and planned destinations

Software and release utilities retain the scoped MIT license. The owner-held rights in the exact 86 fixture/example files and six factual/bibliographic records use CC0-1.0; the exact 13 study/document/chart files use CC BY 4.0. Applied inventories preserve the approved candidate_v4 hashes separately from current exported hashes. These grants do not cover excluded assets or rights the owner does not hold.

The planned repository is [jamesliuai/osu-lab](https://github.com/jamesliuai/osu-lab), downloads will use its versioned GitHub Releases, and the project site is planned at [jamesliuai.github.io/osu-lab](https://jamesliuai.github.io/osu-lab/). These destinations are configured locally; no remote repository, release or Pages deployment has been created by this work. The current release candidate is 0.1.0rc6.
