# Recorded exploratory checks and exposure counts

This is a public derivative of accepted historical internal evidence. It records exploratory and post hoc checks performed outside the pinned analysis tools; it is not a new analysis or review. The preregistered primary result and secondary Holm correction remain the reported confirmatory results.

The accompanying [JSON record](EXPLORATORY_RECEIPT.json) provides semantic fields, exact scoped excerpts and source line ranges. The historical source is `runs/development/pd_20261101/EVIDENCE.md`, SHA-256 `4f51a49c2b877858d12cf624160441f18904c3e40d0d98c23cf697e5c00518a9`. The source file is not included; this is its identity, not a public navigation path. The release registry and manifest record the exported receipt paths and hashes.

Raw per-map batch outputs, test-map manifests and per-update training logs are omitted. Readers can inspect these accepted recorded values, but cannot reconstruct these checks from the omitted raw inputs. No study, audit or training was rerun to prepare this receipt.

## Exploratory error-model and leave-one-replicate checks

The entry dated 2026-11-01T19:58:52-08:00 (lines 453–456) records leave-one-replicate-out slopes of -0.555, -0.878 and -0.836, with one-sided p-values 0.0143,0.0124 and0.0181. Every 90% interval lay beyond the negative margin of -0.15. The excerpt gives the checks in that order without specifying the omitted replicate for each row; this receipt does not assign identities.

With one slope per replicate, the recorded one-sided p-value for H1 is 0.0325. The alternative error-model p-values for S5, S6 and S7 are 0.0011,0.0248 and0.0472; only S5 would remain significant after Holm. These are exploratory checks, not substitutes for the preregistered analysis. The primary p-value and S6 significance depend on the preregistered error term.

## Averaged-policy limitation

The same entry records warm-up `min(d, (1 + t) / (10 + t))` and that the average never reaches its nominal decay 0.9999 during the budget. That entry does not quantify the weight on the last quarter of training or measure whether averaging favors an arm. This receipt does not claim a new check of those report limitations.

## Post hoc spinner counts

The entry dated 2026-11-01T20:12:45-08:00 (lines 491–497) records the following T_ID counts. Spinner results describe the simulator rules; fidelity traces did not validate missed slider ticks, tails or spinner misses against the game engine.

| Run | All failures | Failures on spinner-drill maps | Spinners completed /799 |
|---|---:|---:|---:|
| pd_n16_r1 | 6,036 | 176 | 773 |
| pd_n16_r2 | 4,193 | 343 | 324 |
| pd_n16_r3 | 5,447 | 499 | 167 |
| pd_n256_r1 | 764 | 231 | 411 |
| pd_n256_r2 | 1,206 | 390 | 76 |
| pd_n256_r3 | 1,431 | 412 | 27 |
| pd_n4096_r1 | 274 | 9 | 799 |
| pd_n4096_r2 | 1,068 | 358 | 123 |
| pd_n4096_r3 | 1,108 | 367 | 109 |

Completed spinners fell from 16 to 256 maps and rose from 256 to 4,096 maps in every replicate. Replicate 1 at 4,096 maps alone completed all 799 spinners. Spinner-drill maps account for 349 of its 794-failure advantage over replicate 2 and 358 of 834 over replicate 3: nearly half, with a remaining advantage on other maps. Improvement at 4,096 maps is mostly, not only, through replicate 1. These are post hoc descriptive observations.

## Exposure and repetition

Lines 498–501 of the same entry record these counters.

| Run | Clips per map | Notes per run |
|---|---:|---:|
| pd_n16_r1 | 5,346.6 | 1,655,893 |
| pd_n16_r2 | 5,328.9 | 1,564,556 |
| pd_n16_r3 | 5,538.4 | 1,626,113 |
| pd_n256_r1 | 347.1 | 1,664,125 |
| pd_n256_r2 | 338.7 | 1,642,310 |
| pd_n256_r3 | 341.8 | 1,649,037 |
| pd_n4096_r1 | 21.5 | 1,667,722 |
| pd_n4096_r2 | 21.6 | 1,665,109 |
| pd_n4096_r3 | 21.4 | 1,660,515 |

The recorded notes-per-run span is 6.6%. At a fixed budget, each map received thousands of clips with 16 maps, hundreds with 256 maps and about twenty with 4,096 maps. This design varies diversity and repetition per map together and cannot separate their effects.
