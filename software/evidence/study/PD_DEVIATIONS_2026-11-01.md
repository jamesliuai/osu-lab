# Study PD: deviations from the frozen pre-registration

Dated and labelled, as the frozen pre-registration (`PD_PREREG_frozen_2026-11-01_0223.md`, header and section 8)
requires; the frozen file is not edited. Written 2026-11-01, after the results existed.

## D1. Review rounds beyond three (acted on from 2026-11-01T20:21:32-08:00; recorded here by 2026-11-01T20:25:28-08:00)

- **What.** Section 6 of the pre-registration says: "Review rounds by independent reviewers (...) continue until a
  full round finds no major problem (at most 3 rounds)." The third round on the results document (revision
  `PD_RESULTS_r3.md`) found one major wording problem, so the review continued past three rounds. Revision
  `PD_RESULTS_r4.md` fixed it and was reviewed in a fourth round, which found one major problem: this departure had
  been attributed to the study brief only, and r4 said no deviations file was needed. Revision `PD_RESULTS_r5.md`
  corrects that and is reviewed in a fifth round. The review continues, one full round (all six reviewers) at a time,
  until a full round finds no major problem.
- **Why.** The rule's two parts conflict when the third round finds a major problem. Stopping at three rounds would
  leave a results document with a known major error, while the acceptance condition (a full round with no major
  problem) is what the review is for; the study brief states the same cap (section 10, item 5) and the same acceptance
  condition. Reclassifying a reviewer's major finding to let the third round pass would not be honest.
- **Impact.** None on the design, arms, outcomes, test sets, budget, statistical analysis or any number. The test
  batch ran once; every revision of the results document is checked against the same registered numbers
  (`make_results.py` and `make_results_extra.py check`). Only the wording of the results document changes between
  revisions, and the review record grows by one or more rounds. Logged as executive fix X12 in `EVIDENCE.md`.
- **Timing.** The review is of the results document, so this deviation could not be recorded before the results.

No other deviation from the pre-registration: the design that ran is the pre-registered fallback F1, selected by the
pre-registered rule G0 before the freeze; the analysis script, the outcomes, the test sets, the arms and the budget
are as frozen. The executive fixes that concern tooling, the demo or packaging (X8 to X11) are recorded in
`EVIDENCE.md` and in the results document's Deviations section.

## Addendum to D1 (2026-11-01T20:29:56-08:00): the review ended; clarifications from the fifth round

- **Close.** The fifth full round (all six reviewers, on `PD_RESULTS_r5.md`) found no major problem, so the review
  ended there, about sixteen hours before T_pkg. `PD_RESULTS_r5.md` is the accepted results document. The review was
  open-ended under D1, but it would in any case have stopped at T_pkg, with the latest revision packaged and its open
  findings listed.
- **The rule, precisely.** The pre-registration states a stopping rule with a cap ("continue until a full round finds
  no major problem (at most 3 rounds)") and does not say what happens if the third round is not clean; the
  requirement that a clean full round accept the document comes from the study brief's goal. Stopping at three rounds
  would have left either the known error or an unreviewed fix.
- **Scope of "only the wording".** That applies to the revisions made under D1 (r4 and r5). Revision r2, before D1,
  also added the pre-registered descriptive tables, the plain-language percentages and the labelled exploratory
  error-term paragraph.
- **Why X8 to X11 are not deviations.** None changes anything the pre-registration specifies. X8 uses the demo the
  pre-registration names (section 7). X9 and X11 are tools for the application kit, which the pre-registration does
  not cover. X10 adds a reporting tool next to `make_results.py`: the pre-registered generator and checker are
  unchanged and still produce and check every number they did, and X10 adds the descriptive tables that section 6
  requires, every number still generated from `PD_analysis.json` and re-derived before review.
- **The smoke run.** The short tooling smoke run (launched after the draw through `pd_queue.py`, with a budget of about
  one million decisions and its own `--max-hours 0.3`) comes from the study brief, not the pre-registration; the
  pre-registration's `--max-hours 3.1` applies to the study runs. It read no sealed file and is used for no result.
- **Times of this file.** The first version existed by 20:25:28 (its sha256 then was
  ca386400046dab6b61e274c5635e18c231488f5bd00700f5061df8c3e3c95417); its heading time was then corrected, and the
  file's current creation and write time, 20:25:55, comes from that correction. Both were before the fifth round was
  sent (20:26:05 to 20:26:48).

## Owner decision on D1 (recorded 2026-11-01T21:33:02-08:00)

The owner was asked in the session: "Do you approve deviation D1 (reviewing past three rounds), so PD_RESULTS_r5.md
counts as accepted?" and answered "Approve D1". D1 is approved: `PD_RESULTS_r5.md` is the results document accepted by
the fifth review round (six reviewers, no major finding), with D1 disclosed wherever the review is described.

## Owner's written acceptance (recorded 2026-11-01T21:35:13-08:00, = 2026-11-02T05:35:13Z)

The owner's statement, given in the session on 2026-11-01 (Pacific time):

> I accept D1 retrospectively as a documented deviation from the preregistered review-round limit, and accept
> PD_RESULTS_r5.md as the final internally reviewed results document.

With it the owner asked that the frozen pre-registration, the analysis, the earlier revisions and the review history
be preserved (none of them is edited); that the review procedure not be described as fully compliant with the
original three-round cap and that D1 be retained wherever review acceptance is described; that the pre-registered
analysis be distinguished from exploratory diagnostics; that the checks be described as internal review, without
implying external or human peer review; and that no further full review rounds or experiments be run unless a
specific material defect is found. The completed materials are handed over for release integration; this is not
publication authorization.
