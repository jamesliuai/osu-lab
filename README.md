# osu-lab

Procedural-map reinforcement learning and Study PD, by **James Liu**. Release candidate **0.1.0rc6**.

I developed and integrated a reinforcement-learning system for procedurally generated osu! maps, and designed and ran a study of how practice-map diversity affects generalization. The system adapts upstream osu! and osu-framework rules and uses established PPO and recurrent-network methods.

[Project website](https://jamesliuai.github.io/osu-lab/) · [One-page summary](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/ONE_PAGE_SUMMARY.pdf) · [Technical report](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/REPORT.pdf) · [Release downloads](https://github.com/jamesliuai/osu-lab/releases/tag/v0.1.0rc6)

## Try the CPU example

Use Python 3.12. Clone this repository or extract the prepared source ZIP from the release. In this repository, start in `software/`; in the source ZIP, start in its extracted root.

```sh
cd software
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell), then:

```sh
python -m pip install ".[test]"
python examples/generated_data_example.py --out-dir demo_output
```

The example generates maps and runs scripted on-time and late actions on the CPU. It requires no game client, external assets, checkpoint or GPU. Its judgement counts demonstrate software behavior, not a learned-policy result. Installation fetches dependencies; the example needs no network. The distribution is `osu-sim`; Python imports use `osu_sim`. See [environment and offline installation](software/docs/ENVIRONMENT.md), [reproduction](software/docs/REPRODUCTION.md) and [software documentation](software/README.md).

## Evidence and limitations

Read the [report](software/docs/REPORT.pdf), [summary](software/docs/ONE_PAGE_SUMMARY.pdf), [study results](software/docs/STUDY_RESULTS.md), [scientific limitations](software/docs/LIMITATIONS.md) and [claim registry](software/CLAIM_REGISTRY.json). Results concern generated maps at the recorded training budget. They do not establish performance on human-made maps, human skill or an unlimited scaling law.

The existing internal review exceeded the preregistered three-round cap; the owner approved D1 retrospectively. This remains disclosed and is not external peer review. Publication did not rerun training or reopen scientific review.

The `software/` tree and four release downloads retain the verified candidate bytes. Preparation-status statements inside those preserved documents describe their original preparation, before publication was authorized. Current usage and deployment instructions are in this README and [deployment/README.md](deployment/README.md). Historical verification counts retain their original execution scope.

## Verify the release

[Download metadata](release/DOWNLOAD_MANIFEST.json) and [SHA-256 checksums](release/SHA256SUMS) cover the four named release assets. The prepared source ZIP contains the verified software, documents and evidence; this repository additionally contains the website and publication tooling. GitHub's automatically generated source archives are separate from the prepared `osu-lab-0.1.0rc6-source.zip` asset.

From the repository root, before creating local environments or generated outputs:

```sh
python -I -S -B tools/verify_publication.py --site
python -I -S -B software/tools/verify_release.py --strict
python -I -S -B software/tools/verify_claims.py
python -I -S -B software/tools/verify_documents.py
```

[REPOSITORY_PUBLICATION.json](REPOSITORY_PUBLICATION.json) lists every publication file with its provenance and rights scope. [REPOSITORY_SHA256SUMS](REPOSITORY_SHA256SUMS) covers that inventory and its payload. The strict checks intentionally reject extra generated files. Pages deploys only the eleven files in `site/` through the guarded manual workflow.

## Rights and scope

Software and release utilities retain the scoped MIT license. The exact owner-approved fixture/factual files use CC0-1.0 and the exact study/document/chart files use CC BY 4.0, limited to rights James Liu holds. See [license scopes](LICENSES.md) and [upstream notices](software/THIRD_PARTY_NOTICES.md). Third-party code and font notices remain separate. Game-renderer textures/fonts, uncleared demos, community maps, checkpoints and raw sealed inputs are excluded.

The website includes an owner-approved 33-second video and poster, under the documentation CC BY 4.0 scope for James Liu-held rights. It redraws recorded simulator play with a custom skin; it is separate from the unchanged software release and its prepared downloads. See the website’s Credits and licences section for video context, and its Abstract, Results and Review sections for scientific scope.
