# Assets and license scope

The code license is MIT, matching the existing owner-approved package choice. Upstream-derived portions retain the complete ppy notices in `licenses/`. The release manifest distinguishes unchanged source, portability derivatives, generated fixtures, and new release tooling. James Liu's approved personal-contribution statement is in CONTRIBUTIONS.md; upstream authorship remains separately attributed.

Release-authored documentation follows the package documentation's existing CC BY 4.0 choice; see `LICENSE-DOCS.md`. The exact owner-approved fixture and factual scopes use CC0-1.0 under LICENSE-DATA.md. This sidecar grant does not change frozen records or expand the code license. No weights are distributed.

## Excluded renderer dependencies

`render.py` loads Torus and Venera font descriptors/atlases, FontAwesome glyph resources, and Argon repeat-edge and spinner-glow textures. Those are separate from the MIT implementation. No game texture, game font, atlas, icon binary, or native game resource is included. The new document PDFs separately embed permitted Bitstream Vera subsets, with the exact copyright and license notice attached to each PDF and included in licenses/Bitstream-Vera.txt. The osu-resources repository describes separate resource terms, and Torus/Venera have font-specific licensing conditions. The release does not grant permission for any of them.

Full-renderer screenshots and video may contain those resources, so no such demo is cleared by code or map provenance alone. A generated map does not by itself clear the renderer's assets. Existing videos and old model demos are omitted.

The CPU example emits generated map text and numeric diagnostics. The website architecture diagram uses code-authored vector shapes and generic viewer fonts, with no embedded font binary. It is not gameplay footage. A future full-renderer artifact needs its own source, asset inventory, license/permission decision, and evidence hash before release.

Primary sources: https://github.com/ppy/osu ; https://github.com/ppy/osu-framework ; https://github.com/ppy/osu-resources/blob/master/README.md ; https://github.com/ppy/osu-resources/blob/master/osu.Game.Resources/Fonts/Torus/LICENCE ; https://github.com/ppy/osu-resources/blob/master/osu.Game.Resources/Fonts/Venera/LICENCE . These references describe provenance and separate dependencies, not an asset grant.

## Applied owner-held rights grants

James Liu approved the candidate_v4 inventories: CC0-1.0 for the exact 86 fixture/example files plus six factual/bibliographic records; CC BY 4.0 for the exact 13 study/document/chart files. LICENSE-DATA.md and LICENSE-STUDY.md state the grants. evidence/APPROVED_V4_LICENSE_SCOPE.json pins the approval basis; applied inventories retain each approved hash separately from the current exported hash. Only the four report/summary formats have changed, as attributed public derivatives; all other approved artifacts remain unchanged.

The grants are limited to rights held by James Liu. They do not relicense third-party source, legal texts, trademarks or excluded renderer resources. The study inventory retains the historical filename evidence/study/LICENSE_PROPOSAL.json but its status is owner_approved_applied. No further D1, author, contribution, branding or license approval is pending.

The report and summary are public derivatives, not altered handover originals. Their source identities, changes and semantic evidence appear in evidence/documents/DOCUMENT_PROVENANCE.json. The charts are statistics from the accepted aggregate analysis, not gameplay. Their applied CC BY license does not clear the native renderer's assets.

The separately owned website and Pages deployment files require individual publication inventory entries. They are not automatically included in the 105-file grant. New release-authored repository tooling retains MIT scope, and release-authored explanatory metadata retains its documentation scope; reused website source/assets must retain their own verified provenance and license classification.
