# osu-lab 0.1.0rc6

Prerelease of osu-lab, a procedural-map reinforcement-learning research project by James Liu. The Python distribution is `osu-sim`; imports use `osu_sim`.

This release includes the simulator and procedural generator, CPU examples, retained conformance records, Study PD aggregate evidence, the technical report and one-page summary. The source ZIP and wheel are the exact verified release artifacts. The website is published separately at [jamesliuai.github.io/osu-lab](https://jamesliuai.github.io/osu-lab/).

For the CPU quick-start, use Python 3.12 and follow the [repository README](https://github.com/jamesliuai/osu-lab#try-the-cpu-example). The prepared source ZIP contains the software at its extracted root; the repository stores it under `software/`.

Results are scoped to generated maps, three replicate sets and the recorded training budget. More maps also mean fewer repetitions at fixed exposure. The slope is 3/8 of the 16-versus-4,096 contrast, whose paired-replicate 95% interval includes zero, so the uncertainty depends on the error model. This does not establish performance on human-made maps or an unlimited scaling law. The report preserves confirmatory, descriptive and exploratory distinctions. Internal review exceeded the preregistered three-round cap under retrospectively owner-approved D1; it was not external peer review.

Software and release utilities retain their scoped MIT license and upstream notices. Exact owner-approved fixture/factual files use CC0-1.0; exact study/document/chart files use CC BY 4.0, limited to rights James Liu holds. Third-party notices remain separate. No trained weights, community maps, game-renderer media, raw sealed inputs or original research Git history are included.

## Downloads and SHA-256

GitHub's automatic source archives are distinct from the prepared source ZIP below. [Machine-readable metadata](https://github.com/jamesliuai/osu-lab/blob/v0.1.0rc6/release/DOWNLOAD_MANIFEST.json) and [checksum file](https://github.com/jamesliuai/osu-lab/blob/v0.1.0rc6/release/SHA256SUMS) cover these four assets.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| [osu-lab-0.1.0rc6-source.zip](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/osu-lab-0.1.0rc6-source.zip) | 1,325,167 | `00f686dce2f69a705df21ff4ffbe6c8c426582d65d7d840505b07e1bb0ad94fe` |
| [osu_sim-0.1.0rc6-py3-none-any.whl](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/osu_sim-0.1.0rc6-py3-none-any.whl) | 171,567 | `f24a3f83b8497b6e2937234295887ece59b149aa7d08bea9dad03a7ec443f01a` |
| [REPORT.pdf](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/REPORT.pdf) | 457,093 | `de75466148c32973be749c263939ad88b06c40a0dd0f07543befc87d869f4358` |
| [ONE_PAGE_SUMMARY.pdf](https://github.com/jamesliuai/osu-lab/releases/download/v0.1.0rc6/ONE_PAGE_SUMMARY.pdf) | 174,069 | `7d257717a8ee1106ccd1b7b40f475f22a320ebf1b86941fa8df6a5222712ab3f` |
