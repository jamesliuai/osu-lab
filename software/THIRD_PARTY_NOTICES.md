# Third-party notices for osu-lab

This package includes Python adaptations of behavior and algorithms from osu!lazer and osu-framework. Those portions are covered by the ppy MIT notices reproduced in `licenses/osu-LICENCE.txt` and `licenses/osu-framework-LICENCE.txt`; retain both notices with redistributed copies or substantial portions.

Upstream-derived portions include beatmap decoding and conversion, slider paths and nested events, stacking, difficulty mappings, judgement and rate handling, and portions of visual transform, text layout and color behavior. Relevant modules are `osu_sim/beatmap.py`, `difficulty.py`, `osufile.py`, `rules.py`, `rate_rules.py`, and `render.py`. The source identifies pinned osu commit `aaa75b7c` and framework commit `247dee98`. These are adaptations rather than unmodified copies of the complete upstream projects.

The project's engineering includes the vectorized simulator interface, procedural map generation, policy and training integration, render-on-demand rollout machinery, evaluation tools, and generated-content regression checks. This description does not claim invention of upstream game rules, PPO, recurrent networks, or standard mathematical methods.

The calibrated renderer's external textures, font descriptors and atlases are not distributed. They have separate terms from the source code. No permission for those assets, community maps, music, osu! or ppy branding is granted by this package's code license. The project is not represented as an official or endorsed osu! product.

Installed dependencies are obtained separately and retain their own licenses. This source release does not redistribute NumPy, Numba, llvmlite, PyTorch, Triton, OpenCV, PyYAML or pytest binaries.

Upstream sources: [osu!](https://github.com/ppy/osu), [osu-framework](https://github.com/ppy/osu-framework), and the separate [osu-resources license information](https://github.com/ppy/osu-resources/blob/master/README.md).

The public report and summary PDFs embed subsets of Bitstream Vera, copyright (c) 2003 Bitstream, Inc.; Bitstream Vera is a trademark of Bitstream, Inc. The exact copyright and permission notice is included in licenses/Bitstream-Vera.txt and attached to each standalone PDF. These document fonts are separate from the excluded game-renderer resources. Charts use ordinary vector shapes and generic font references; their PNG versions contain raster text, not distributed game fonts or textures.

The bibliography cites external scholarly works. No cited paper, abstract, publisher image or supplementary asset is redistributed. Individual metadata checks and bounded access limits appear in evidence/documents/REFERENCE_AUDIT.json.

The public project/repository name is osu-lab. The software distribution osu-sim and Python import names are unchanged. James Liu's approved personal-contribution statement appears in docs/CONTRIBUTIONS.md; it does not replace or diminish upstream attribution.

The owner's CC0 and CC BY grants apply only to the exact candidate_v4-derived scopes identified by LICENSE-DATA.md and LICENSE-STUDY.md. Neither grant relicenses the adapted code, the ppy notices, third-party legal texts, font software or excluded renderer assets. Complete CC legal texts are reproduced unchanged in licenses/CC0-1.0.txt and licenses/CC-BY-4.0.txt.
