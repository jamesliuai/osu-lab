# Limitations

- Generated examples come from one procedural generator. No claim about community maps or human play is supported.
- Absolute cursor commands remove a major human motor constraint; synthetic difficulty labels are not native star ratings.
- Generated conformance is finite and has missing failure cases. The retained records are insufficient for an independent native per-step replay.
- The calibrated renderer is not pixel-exact. Its textures and font atlases are excluded, and it was not exercised for this release. The simple renderer is a diagnostic approximation.
- The trainer targets CUDA. CPU examples validate generation and simulation, not learning or trained policy quality. No GPU work was started for release preparation.
- Release portability edits change path and asset discovery and some explanatory text. This is a reviewed derivative of selected sources, not a byte-identical frozen study environment. All frozen fixtures remain byte-identical.
- The accepted F1 study is included as aggregate evidence with its conditional error model and D1 review-cap deviation disclosed in STUDY_RESULTS.md. No weights, raw sealed evaluations, gameplay footage or frozen training environment are distributed.
- Installed third-party dependencies and renderer assets retain their own terms. The package license grants no rights in osu!/ppy names, community maps, music, or external assets.
