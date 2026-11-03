"""Portable derivative of the Study PD fallback F1 analysis.

This exported derivative is not a frozen preregistration or the byte-identical study analysis source. Statistical
functions and F1 design constants are preserved. Documentation corrects main-design descriptions left in the original
F1 file. Root inference is removed, script provenance uses its filename and hash, and --root is explicit for analysis.
Requires NumPy and SciPy; CPU only. It evaluates no policy and opens no pool (.npz). Inputs are a caller-supplied batch
summary, listed evaluation JSONs, test/probe manifests and optional learning-curve JSONL files. Supply only trusted
input bundles: listed absolute paths are honored, and every file read is recorded with its hash.

    python -B tools/pd_analysis_f1.py --batch BATCH.json --root INPUT_ROOT --out-json analysis.json --out-txt analysis.txt
    python -B tools/pd_analysis_f1.py --power-table

Outputs are created exclusively. Relative input paths resolve against required --root; command-line batch/output
paths resolve normally. No original repository location is inferred. Synthetic demonstration inputs and outputs are
clearly marked by tools/synthetic_analysis_example.py and are never evidence about trained agents.

DEFINITIONS (labelled; fixed before any PD result)
P1 F1 design. Arms N in {16, 256, 4096} distinct generated practice maps; replicates r in {1, 2, 3}; k = log8(N /
   16), with values 0, 4/3, 8/3. Each run's final EMA-0.9999 policy played (deterministic) ONE merged test pool
   whose manifest gives every map's set: "id" (500 maps of the practice construction), "tempo" (250 maps at 1.3x tempo)
   and "pattern" (250 burst drills); and, separately, its train probe (its first min(N, 256) practice maps). Map records
   = detail.deterministic.maps of an eval JSON, one per map, in pool order.
P2 Outcome of a set of map records: F = sum of (breaks + tail_miss), O = sum of max_combo_possible, Y = ln(F / O) (lower
   is better); zero-failure rule: F == 0 -> Y = ln(0.5 / O); O == 0 -> Y null. fc_rate = mean of the per-map "fc",
   mean_acc = mean of the per-map "acc" (both unweighted over maps), n_maps = number of map records. Per run: Y_id,
   Y_tempo and Y_pattern (test eval, maps grouped by the manifest's set), Y_probe (every map of the probe eval) and the
   memorization gap G = Y_id - Y_probe.
P3 Complete-set rule. A replicate counts only if all three of its arms have a run record with complete == true (the JSON
   boolean). Only complete replicates enter the primary, the secondaries and the per-arm descriptives; every other run
   is listed in excluded_runs with its reason. With fewer than 3 complete replicates (brief section 12 F3) the primary
   and every secondary are "Not tested: insufficient replicate sets" (no estimate, interval or p-value); the
   descriptive results are still computed and reported.
P4 Primary (H1: b < 0). OLS Y_id = alpha_r + b k + e over the n = 3R runs of the R complete replicates, one intercept
   per replicate (replicate fixed effects, no common intercept). Classical OLS standard error se = s / sqrt(Sxx) with
   s^2 = RSS / df, df = n - R - 1 and Sxx = the sum of squared within-replicate deviations of k (32/9 per complete
   replicate); t = b / se; one-sided p = P(T_df <= t); 95% CI b +- t(0.975, df) se; 90% CI b +- t(0.95, df) se;
   residual SD = s; intercepts alpha_r. If se == 0: p = 0 when b < 0, else 1 (t is then not finite: null).
P5 R5 category, with delta = 0.15 (ln units per 8x step of N), p_adj = the primary's one-sided p (the primary family has
   one test) and the 90% CI [lo, hi]: inside = lo > -delta and hi < +delta; beyond = lo > +delta or hi < -delta (all
   strict). p < 0.05 and beyond: "Exceeds the margin"; p < 0.05 and inside: "Detectable but within the margin";
   p < 0.05 and neither: "Detectable difference, magnitude uncertain"; p >= 0.05 and inside: "Equivalent within the
   margin"; p >= 0.05 and not inside: "Inconclusive". practically_meaningful = (category == "Exceeds the margin"); no
   other category is given that label in any output.
P6 Secondary family: 6 tests, each one-sided in the predicted direction (estimate < 0), Holm over m = 6 at alpha 0.05.
   S1-S3 are paired over the complete replicates: d_r = Y_id(4096) - Y_id(16) (S1), Y_id(256) - Y_id(16) (S2),
   Y_id(4096) - Y_id(256) (S3); estimate = mean d, se = sd / sqrt(n) (sd with ddof 1),
   df = n - 1, t = mean / se, p = P(T_df <= t), 95% CI mean +- t(0.975, df) se (se == 0 as in P4). S5-S7 are the P4
   model's slope on G (S5), Y_tempo (S6) and Y_pattern (S7). Holm: p sorted ascending, p_holm_(i) = max over j <= i of
   min(1, (m - j + 1) p_(j)); reject when p_holm < 0.05.
P7 Descriptive (no tests). by_arm: per arm over the complete replicates, the mean, the 95% t-interval (df = n - 1; null
   when n < 2), n and the per-replicate values (values_by_replicate) of Y_id, Y_probe, G, Y_tempo, Y_pattern,
   fc_rate_id and mean_acc_id.
   by_band / by_variant: per arm, the id set pooled over the complete replicates by the manifest's band (0, 1, 2) and
   variant (slider, spinner, streamjump, plain): F, O and Y (P2 on the pooled counts). references: keyed by the name
   of each reference policy in the batch summary, {id, tempo, pattern (the P2 values of each set), kind, note}.
   learning_curves: keyed by run name, every run record (complete or not), {N, replicate, monitor, probe, note}:
   monitor = the eval.jsonl rows named "pd_monitor_ema", probe = the rows named "pd_probe_ema", each row ->
   [decisions, Y, mean_acc, fc_rate] (settings.curve_columns) from the row's summary.deterministic (failures and
   objects with the P2 rules); a missing file is skipped with a note.
P8 Integrity; a failure refuses (SystemExit) before anything is written: the batch summary's schema is
   "pd_batch_summary_v1"; the test manifest's sha256 equals its record; every eval file's sha256 equals its record;
   every test eval's per-map "map" keys equal the test manifest's key list exactly (same maps, same order: every policy
   played the identical merged pool), references included; every probe eval's keys equal its probe manifest's keys; no
   duplicate (N, replicate).
P9 Power (preregistered; power_table(), --power-table): for R = 3 complete replicates and residual SD sigma in
   (0.05, 0.10, 0.15, 0.20, 0.30, 0.40): df = 3R - R - 1, Sxx = (32/9)R, se = sigma / sqrt(Sxx), 95% CI half-width
   t(0.975, df) se, 90% CI half-width t(0.95, df) se, approximate minimum detectable slope (one-sided alpha 0.05, 80%
   power) (t(0.95, df) + t(0.80, df)) se.
P10 Wording. wording.primary_sentence states the estimate, its 95% CI, the one-sided p and the R5 category; only the
   category "Exceeds the margin" is worded as practically_meaningful. No output contains a word from FORBIDDEN
   (checked before anything is written).
LABELLED CHOICES (left open by the brief; fixed here before any result)
C1 A run record with complete == true must name a test_eval and a probe_eval (with sha256 records) and a
   probe_manifest, else refuse: a batch fault, not a design exclusion. An incomplete run's evals are read and checked
   when present; its per-run values are reported in "runs" and never pooled; absent evals -> null sets.
C2 Refused as malformed: N outside the three F1 arms, a replicate outside 1-3, a run without a name, duplicate run names,
   duplicate reference names; a test-manifest map whose set is not id / tempo / pattern, a set with no map, duplicate
   manifest keys; a map record without map, fc, max_combo_possible, breaks, tail_miss or acc, or with a non-numeric
   count. sha256 records compare case-insensitively.
C3 A secondary test that is not estimable (a null per-run value) enters Holm as p = 1 (m stays 6); it reports status
   "Not estimable: ..." and null values. A test that runs has status "Tested". Not tested -> every value null, n null.
   The primary reports n (runs in complete replicate sets) and n_replicates in every status.
C4 by_band / by_variant cells also give n_maps (that cell's test maps, per run) and n_runs (complete replicates pooled);
   with no complete replicate F, O and Y are null.
C5 Learning-curve rows keep file order (duplicates kept); a row without failures or objects gets Y null; an unparsable
   line is skipped and counted in the note; eval_jsonl null or a missing file -> empty curves and a note.
C6 "version" is an object: the version label, this script's path and sha256, and the python, numpy and scipy versions.
C7 Recorded paths are relative to --root (posix) when the file lies under it, else absolute.
C8 A reference whose test_eval is null gets id, tempo and pattern null and a note.
C9 A reference's test_eval may instead be the <out>_maps.json of scripts/sim_autopilot_objects.py (the brief's tool for
   the autopilot): the same per-map records as a top-level "maps" list, with the same sha256 and key checks. Run evals
   must be scripts/sim_eval_pool.py output (detail.deterministic.maps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

HERE = Path(__file__).resolve()

VERSION = "pd_analysis F1 portable release derivative (2026-11-01)"
SCHEMA_IN = "pd_batch_summary_v1"
SCHEMA_OUT = "pd_analysis_result_v1"
ARMS = (16, 256, 4096)
K_OF = {16: 0.0, 256: 4 / 3, 4096: 8 / 3}             # k = log8(N / 16) under fallback F1 (P1)
REPLICATES = (1, 2, 3)
TEST_SETS = ("id", "tempo", "pattern")
MODE = "deterministic"
DELTA = 0.15
ALPHA = 0.05
MIN_COMPLETE = 3
SXX_PER_REPLICATE = 32 / 9                             # sum over k = 0, 4/3, 8/3 of (k - 4/3)^2 (P4, P9)
POWER_R = (3,)
POWER_SIGMAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
BANDS = (0, 1, 2)
VARIANTS = ("slider", "spinner", "streamjump", "plain")
MAP_FIELDS = ("map", "fc", "max_combo_possible", "breaks", "tail_miss", "acc")
CURVES = (("monitor", "pd_monitor_ema"), ("probe", "pd_probe_ema"))    # learning-curve key <- eval.jsonl row name
CURVE_NAMES = tuple(row_name for _, row_name in CURVES)
CURVE_COLUMNS = ("decisions", "Y", "mean_acc", "fc_rate")
PER_RUN_METRICS = ("Y_id", "Y_probe", "G", "Y_tempo", "Y_pattern", "fc_rate_id", "mean_acc_id")
TESTED = "Tested"
NOT_TESTED = "Not tested: insufficient replicate sets"
EXCEEDS = "Exceeds the margin"
DETECTABLE_UNCERTAIN = "Detectable difference, magnitude uncertain"
DETECTABLE_WITHIN = "Detectable but within the margin"
EQUIVALENT = "Equivalent within the margin"
INCONCLUSIVE = "Inconclusive"
CATEGORIES = (EXCEEDS, DETECTABLE_UNCERTAIN, DETECTABLE_WITHIN, EQUIVALENT, INCONCLUSIVE)
SECONDARY = (                                          # (id, kind, outcome, (arm a, arm b) for a - b, label) (P6)
    ("S1", "paired", "Y_id", (4096, 16), "Y_id(4096) - Y_id(16), paired by replicate"),
    ("S2", "paired", "Y_id", (256, 16), "Y_id(256) - Y_id(16), paired by replicate"),
    ("S3", "paired", "Y_id", (4096, 256), "Y_id(4096) - Y_id(256), paired by replicate"),
    ("S5", "slope", "G", None, "slope of the memorization gap G = Y_id - Y_probe on k, replicate fixed effects"),
    ("S6", "slope", "Y_tempo", None, "slope of Y_tempo on k, replicate fixed effects"),
    ("S7", "slope", "Y_pattern", None, "slope of Y_pattern on k, replicate fixed effects"),
)
TOP_KEYS = ("schema", "version", "inputs", "settings", "complete_replicates", "excluded_runs", "runs", "primary",
            "secondary", "descriptive", "power", "wording")
FORBIDDEN = (r"\bunstudied\b", r"\bsuperhuman\b", r"\bproves\b")
MEANINGFUL = r"practically\s+meaningful"


def _labels(doc):
    """The labelled definitions and choices of the module docstring, one string per label (copied into the outputs)."""
    body = (doc or "").split("DEFINITIONS (labelled; fixed before any PD result)", 1)[-1]
    out = []
    for line in body.split("\n"):
        s = line.strip()
        if not s or s.startswith("LABELLED CHOICES"):
            continue
        if re.match(r"^[PC]\d+ ", s):
            out.append(s)
        elif out:
            out[-1] += " " + s
    return out


LABELS = _labels(__doc__)


# ========================================================================================================== utilities
def refuse(msg):
    """P8 / C1 / C2 refusal: SystemExit with a message (raised before any output is written)."""
    raise SystemExit(f"pd_analysis: REFUSED: {msg}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ledger:
    """Every file read, with its sha256 (the JSON's "inputs"). Relative paths resolve against the root (C7)."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def full(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def record_path(self, full):
        full = Path(full).resolve()
        try:
            return full.relative_to(self.root).as_posix()
        except ValueError:
            return full.as_posix()

    def read(self, p, role, where=None):
        full = self.full(p)
        if not full.is_file():
            refuse(f"{where or role}: file not found: {p}")
        data = full.read_bytes()
        sha = sha256_bytes(data)
        key = self.record_path(full)
        prev = self.files.get(key)
        if prev is None:
            self.files[key] = dict(path=key, sha256=sha, role=role)
        elif prev["sha256"] != sha:
            refuse(f"{key} changed between two reads")
        return data, sha

    def manifest(self):
        return [self.files[k] for k in sorted(self.files)]


def parse_json(data, where):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        refuse(f"{where}: not valid UTF-8 JSON ({type(e).__name__})")


def read_json(led, path, role, where):
    data, _ = led.read(path, role, where)
    return parse_json(data, where)


def read_checked(led, path, sha_record, role, where):
    """Read a file whose sha256 the batch summary records; refuse on a mismatch (P8; case-insensitive, C2)."""
    data, sha = led.read(path, role, where)
    rec = sha_record.strip().lower() if isinstance(sha_record, str) else None
    if rec != sha:
        refuse(f"{where}: sha256 of {path} is {sha}; the batch summary records {sha_record!r}")
    return parse_json(data, where)


def clean(obj):
    """JSON-safe copy: numpy scalars -> python, NaN / inf -> None, tuples -> lists, dict keys -> str."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def fnum(x, nd=4):
    """A JSON value for the text: null -> n/a, booleans -> true / false, integers as they are, floats to nd decimals."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"
    return str(x)


def fp(p):
    """A p-value for the text: 4 decimals, or 3 significant digits below 1e-4."""
    if p is None or isinstance(p, bool) or not isinstance(p, (int, float)):
        return "n/a"
    p = float(p)
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def fci(ci, nd=4):
    if not ci or ci[0] is None or ci[1] is None:
        return "[n/a]"
    return f"[{fnum(ci[0], nd)}, {fnum(ci[1], nd)}]"


# ============================================================================================================ outcome
def y_of(F, O):
    """P2: ln(F / O); zero-failure rule F == 0 -> ln(0.5 / O); O <= 0 (or a missing count) -> None."""
    if F is None or O is None or O <= 0 or F < 0:
        return None
    return math.log(F / O) if F > 0 else math.log(0.5 / O)


def set_outcome(maps):
    """P2 outcome of a list of per-map records -> {F, O, Y, fc_rate, mean_acc, n_maps}."""
    n = len(maps)
    F = sum(int(m["breaks"]) + int(m["tail_miss"]) for m in maps)
    O = sum(int(m["max_combo_possible"]) for m in maps)
    return dict(F=F, O=O, Y=y_of(F, O),
                fc_rate=(math.fsum(1.0 if m["fc"] else 0.0 for m in maps) / n) if n else None,
                mean_acc=(math.fsum(float(m["acc"]) for m in maps) / n) if n else None, n_maps=n)


# ============================================================================================================== stats
def one_sided_lower(est, se, df):
    """t = est / se and the one-sided p of H1 'estimate < 0', P(T_df <= t) (P4). se == 0: p = 0 if est < 0, else 1
    (t = -inf / +inf / nan)."""
    if se > 0:
        t = est / se
        return t, float(stats.t.cdf(t, df))
    if est < 0:
        return -math.inf, 0.0
    if est > 0:
        return math.inf, 1.0
    return math.nan, 1.0


def ols_fe(y, k, groups):
    """P4: OLS of y on k with one intercept per group (replicate fixed effects) and the classical standard error.

    Returns {estimable, n, n_groups, df, b, se, t, p_one_sided, ci95, ci90, residual_sd, intercepts, sxx, rss}
    (intercepts keyed by str(group)); estimable False with a reason when df < 1 or k does not vary within groups."""
    y = [float(v) for v in y]
    k = [float(v) for v in k]
    g = list(groups)
    if not len(y) == len(k) == len(g):
        raise ValueError("y, k and groups differ in length")
    n = len(y)
    labels = sorted(set(g))
    df = n - len(labels) - 1
    base = dict(n=n, n_groups=len(labels), df=df)
    if df < 1:
        return dict(base, estimable=False, reason=f"residual df {df} < 1")
    members = {lab: [i for i in range(n) if g[i] == lab] for lab in labels}
    kbar = {lab: math.fsum(k[i] for i in members[lab]) / len(members[lab]) for lab in labels}
    ybar = {lab: math.fsum(y[i] for i in members[lab]) / len(members[lab]) for lab in labels}
    kd = [k[i] - kbar[g[i]] for i in range(n)]
    yd = [y[i] - ybar[g[i]] for i in range(n)]
    sxx = math.fsum(v * v for v in kd)
    if not sxx > 0:
        return dict(base, estimable=False, reason="k does not vary within replicates")
    b = math.fsum(a * c for a, c in zip(kd, yd)) / sxx
    alpha = {lab: ybar[lab] - b * kbar[lab] for lab in labels}
    resid = [y[i] - alpha[g[i]] - b * k[i] for i in range(n)]
    rss = math.fsum(e * e for e in resid)
    s = math.sqrt(rss / df)
    se = s / math.sqrt(sxx)
    t, p = one_sided_lower(b, se, df)
    q975, q95 = float(stats.t.ppf(0.975, df)), float(stats.t.ppf(0.95, df))
    return dict(base, estimable=True, b=b, se=se, t=t, p_one_sided=p, ci95=[b - q975 * se, b + q975 * se],
                ci90=[b - q95 * se, b + q95 * se], residual_sd=s,
                intercepts={str(lab): alpha[lab] for lab in labels}, sxx=sxx, rss=rss)


def paired_test(d):
    """P6 paired one-sided t of H1 'mean d < 0' -> {estimable, n, df, estimate, sd, se, t, p_one_sided, ci95}."""
    d = [float(x) for x in d]
    n = len(d)
    if n < 2:
        return dict(estimable=False, reason="fewer than 2 pairs", n=n)
    mean = math.fsum(d) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t, p = one_sided_lower(mean, se, df)
    q975 = float(stats.t.ppf(0.975, df))
    return dict(estimable=True, n=n, df=df, estimate=mean, sd=sd, se=se, t=t, p_one_sided=p,
                ci95=[mean - q975 * se, mean + q975 * se])


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order (P6)."""
    p = [float(x) for x in pvals]
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def r5_category(p_adj, ci90, delta=DELTA, alpha=ALPHA):
    """P5: the R5 category of a test from its (adjusted) p and 90% CI; None when either is missing."""
    if p_adj is None or ci90 is None or ci90[0] is None or ci90[1] is None:
        return None
    lo, hi = ci90
    inside = (lo > -delta) and (hi < delta)
    beyond = (lo > delta) or (hi < -delta)
    if p_adj < alpha:
        if beyond:
            return EXCEEDS
        if inside:
            return DETECTABLE_WITHIN
        return DETECTABLE_UNCERTAIN
    return EQUIVALENT if inside else INCONCLUSIVE


def describe(values):
    """P7: n, mean and 95% t-interval (df = n - 1) of the non-null values; ci95 None when n < 2."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    n = len(v)
    if n == 0:
        return dict(n=0, mean=None, ci95=None)
    mean = math.fsum(v) / n
    if n < 2:
        return dict(n=n, mean=mean, ci95=None)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in v) / (n - 1))
    h = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, ci95=[mean - h, mean + h])


def power_table():
    """P9: the preregistered precision / power table of the primary slope (no inputs)."""
    rows = []
    for R in POWER_R:
        n = len(ARMS) * R
        df = n - R - 1
        sxx = SXX_PER_REPLICATE * R
        q975, q95, q80 = (float(stats.t.ppf(q, df)) for q in (0.975, 0.95, 0.80))
        for sigma in POWER_SIGMAS:
            se = sigma / math.sqrt(sxx)
            rows.append(dict(R=R, n=n, sigma=sigma, df=df, sxx=sxx, se=se, ci95_half_width=q975 * se,
                             ci90_half_width=q95 * se, min_detectable_one_sided=(q95 + q80) * se))
    return rows


def power_table_text(rows):
    L = ["PD F1 primary power table (P9): OLS slope b of Y_id on k, replicate fixed "
         "effects, 3 arms per complete replicate set",
         "min_detectable_one_sided = (t(0.95, df) + t(0.80, df)) * se: approximate, one-sided alpha 0.05, 80% power",
         f"{'R':>2} {'n':>3} {'df':>3} {'sigma':>6} {'Sxx':>4} {'se':>8} {'ci95_half_width':>16} "
         f"{'ci90_half_width':>16} {'min_detectable_one_sided':>25}"]
    for r in rows:
        L.append(f"{r['R']:>2} {r['n']:>3} {r['df']:>3} {r['sigma']:>6.2f} {r['sxx']:>4} {r['se']:>8.4f} "
                 f"{r['ci95_half_width']:>16.4f} {r['ci90_half_width']:>16.4f} {r['min_detectable_one_sided']:>25.4f}")
    return "\n".join(L) + "\n"


# ============================================================================================================= inputs
def parse_test_manifest(obj):
    """P1 / C2: (keys, set, band, variant) per map of the merged test manifest, in pool order."""
    maps = obj.get("maps") if isinstance(obj, dict) else None
    if not isinstance(maps, list) or not maps:
        refuse("test manifest: no 'maps' list")
    keys, sets, bands, variants = [], [], [], []
    for i, m in enumerate(maps):
        if not isinstance(m, dict) or "key" not in m:
            refuse(f"test manifest: map {i} has no key")
        if m.get("set") not in TEST_SETS:
            refuse(f"test manifest: map {i} has set {m.get('set')!r}, not one of {list(TEST_SETS)}")
        keys.append(str(m["key"]))
        sets.append(m["set"])
        bands.append(m.get("band"))
        variants.append(m.get("variant"))
    if len(set(keys)) != len(keys):
        refuse("test manifest: duplicate map keys")
    for s in TEST_SETS:
        if s not in sets:
            refuse(f"test manifest: set {s!r} has no map")
    return keys, sets, bands, variants


def manifest_keys(obj, where):
    """The key list of a probe manifest, in pool order (P8)."""
    maps = obj.get("maps") if isinstance(obj, dict) else None
    if not isinstance(maps, list) or not maps:
        refuse(f"{where}: no 'maps' list")
    keys = []
    for i, m in enumerate(maps):
        if not isinstance(m, dict) or "key" not in m:
            refuse(f"{where}: map {i} has no key")
        keys.append(str(m["key"]))
    if len(set(keys)) != len(keys):
        refuse(f"{where}: duplicate map keys")
    return keys


def eval_maps(obj, where, allow_maps_json=False):
    """The per-map records of an eval JSON (detail.deterministic.maps), checked for the fields P2 uses (C2). With
    allow_maps_json (references only, C9) a file without that path may give them as a top-level "maps" list."""
    try:
        maps = obj["detail"][MODE]["maps"]
    except (KeyError, TypeError):
        maps = obj.get("maps") if allow_maps_json and isinstance(obj, dict) else None
        if maps is None:
            refuse(f"{where}: no detail.{MODE}.maps" + (" and no top-level maps list" if allow_maps_json else ""))
    if not isinstance(maps, list) or not maps:
        refuse(f"{where}: the per-map records are empty or not a list")
    for i, m in enumerate(maps):
        if not isinstance(m, dict):
            refuse(f"{where}: map record {i} is not an object")
        missing = [f for f in MAP_FIELDS if f not in m]
        if missing:
            refuse(f"{where}: map record {i} lacks {missing}")
        try:
            int(m["breaks"]), int(m["tail_miss"]), int(m["max_combo_possible"]), float(m["acc"])
        except (TypeError, ValueError):
            refuse(f"{where}: map record {i} has a non-numeric count or accuracy")
    return maps


def check_keys(maps, keys, where, against):
    """P8: the records' "map" keys equal the manifest's key list exactly (same maps, same order)."""
    got = [str(m["map"]) for m in maps]
    if got == keys:
        return
    if len(got) == len(keys) and sorted(got) == sorted(keys):
        first = next(i for i, (a, b) in enumerate(zip(got, keys)) if a != b)
        refuse(f"{where}: the same maps as {against} in a different order (first difference at position {first}); "
               "every policy must play the identical pool")
    refuse(f"{where}: per-map keys differ from {against} ({len(got)} records vs {len(keys)} keys; "
           f"{len(set(keys) - set(got))} missing, {len(set(got) - set(keys))} unexpected)")


def _as_int(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return int(v) if float(v).is_integer() else None


def rec_key(rec):
    return _as_int(rec.get("N")), _as_int(rec.get("replicate"))


def run_order(rec):
    N, r = rec_key(rec)
    return r, ARMS.index(N)


def validate_runs(records):
    """C2 / P8: names, arms, replicates; no duplicate run name and no duplicate (N, replicate)."""
    if not isinstance(records, list) or not records:
        refuse("batch summary: 'runs' must be a non-empty list")
    seen, names = set(), set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            refuse(f"batch summary: run record {i} is not an object")
        name = rec.get("name")
        if not isinstance(name, str) or not name:
            refuse(f"batch summary: run record {i} has no name")
        if name in names:
            refuse(f"batch summary: duplicate run name {name!r}")
        names.add(name)
        N, r = rec_key(rec)
        if N not in ARMS:
            refuse(f"run {name}: N = {rec.get('N')!r} is not one of the arms {list(ARMS)}")
        if r not in REPLICATES:
            refuse(f"run {name}: replicate {rec.get('replicate')!r} is not one of {list(REPLICATES)}")
        if (N, r) in seen:
            refuse(f"duplicate (N, replicate) = ({N}, {r}) (run {name})")
        seen.add((N, r))


def process_run(led, rec, keys, set_of):
    """P2 per-run outcomes with the P8 / C1 checks -> (runs entry, the test eval's map records or None)."""
    name = rec["name"]
    N, r = rec_key(rec)
    complete = rec.get("complete") is True
    test_maps = probe_maps = None
    if rec.get("test_eval"):
        where = f"run {name} test_eval"
        test_maps = eval_maps(read_checked(led, rec["test_eval"], rec.get("test_eval_sha256"), "test_eval", where),
                              where)
        check_keys(test_maps, keys, where, "the test manifest")
    elif complete:
        refuse(f"run {name} is complete but has no test_eval (C1)")
    if rec.get("probe_eval"):
        where = f"run {name} probe_eval"
        if not rec.get("probe_manifest"):
            refuse(f"{where}: no probe_manifest to check its keys against (C1)")
        probe_maps = eval_maps(read_checked(led, rec["probe_eval"], rec.get("probe_eval_sha256"), "probe_eval",
                                            where), where)
        pw = f"run {name} probe_manifest"
        check_keys(probe_maps, manifest_keys(read_json(led, rec["probe_manifest"], "probe_manifest", pw), pw), where,
                   "its probe manifest")
    elif complete:
        refuse(f"run {name} is complete but has no probe_eval (C1)")
    sets = {s: (set_outcome([m for m, t in zip(test_maps, set_of) if t == s]) if test_maps is not None else None)
            for s in TEST_SETS}
    sets["probe"] = set_outcome(probe_maps) if probe_maps is not None else None
    y_id = sets["id"]["Y"] if sets["id"] is not None else None
    y_pr = sets["probe"]["Y"] if sets["probe"] is not None else None
    G = (y_id - y_pr) if y_id is not None and y_pr is not None else None
    return dict(name=name, N=N, replicate=r, k=K_OF[N], complete=complete, sets=sets, G=G), test_maps


def complete_replicates(records):
    """P3: (complete replicates, excluded runs [{name, reason}]) from the batch summary's run records."""
    by = {rec_key(rec): rec for rec in records}
    complete, problems = [], {}
    for r in REPLICATES:
        bad = []
        for N in ARMS:
            rec = by.get((N, r))
            if rec is None:
                bad.append(f"N={N}: no run record")
            elif rec.get("complete") is not True:
                bad.append(f"N={N}: complete is not true")
        if bad:
            problems[r] = bad
        else:
            complete.append(r)
    excluded = []
    for rec in sorted(records, key=run_order):
        N, r = rec_key(rec)
        if rec.get("complete") is not True:
            detail = rec.get("complete_detail") or "no complete_detail"
            excluded.append(dict(name=rec["name"], reason=f"complete is not true ({detail}); replicate {r} is not "
                                                          "a complete set"))
        elif r not in complete:
            excluded.append(dict(name=rec["name"], reason=f"replicate {r} is not a complete set: "
                                                          + "; ".join(problems[r])))
    return complete, excluded


# =========================================================================================================== analysis
def run_values(entry):
    """The P7 per-run metrics of a runs entry."""
    s = entry["sets"]

    def get(name, field):
        return s[name][field] if s.get(name) is not None else None

    return dict(Y_id=get("id", "Y"), Y_probe=get("probe", "Y"), G=entry["G"], Y_tempo=get("tempo", "Y"),
                Y_pattern=get("pattern", "Y"), fc_rate_id=get("id", "fc_rate"), mean_acc_id=get("id", "mean_acc"))


def slope_fit(complete, values, outcome):
    """P4 model on one outcome over the complete replicates."""
    y, k, g, missing = [], [], [], []
    for r in complete:
        for N in ARMS:
            v = values[(N, r)][outcome]
            if v is None:
                missing.append(f"N={N} r={r}")
            y.append(v)
            k.append(K_OF[N])
            g.append(r)
    if missing:
        return dict(estimable=False, reason=f"{outcome} is null for {', '.join(missing)}")
    return ols_fe(y, k, g)


def paired_fit(complete, values, outcome, pair):
    """P6 paired test of outcome(arm a) - outcome(arm b) over the complete replicates."""
    a, b = pair
    d, missing = [], []
    for r in complete:
        va, vb = values[(a, r)][outcome], values[(b, r)][outcome]
        if va is None or vb is None:
            missing.append(f"r={r}")
        else:
            d.append(va - vb)
    if missing:
        return dict(estimable=False, reason=f"{outcome} is null for {', '.join(missing)}")
    return paired_test(d)


def primary_block(complete, values):
    """P3-P5 primary record (keys fixed by the brief)."""
    R = len(complete)
    out = dict(status=None, outcome="Y_id", n=len(ARMS) * R, n_replicates=R, df=None, b=None, se=None, t=None,
               p_one_sided=None, ci95=None, ci90=None, residual_sd=None, intercepts=None, delta=DELTA,
               r5_category=None, practically_meaningful=None)
    if R < MIN_COMPLETE:
        out["status"] = NOT_TESTED
        return out
    fit = slope_fit(complete, values, "Y_id")
    if not fit["estimable"]:
        out["status"] = f"Not estimable: {fit['reason']}"
        return out
    cat = r5_category(fit["p_one_sided"], fit["ci90"])
    out.update(status=TESTED, n=fit["n"], df=fit["df"], b=fit["b"], se=fit["se"], t=fit["t"],
               p_one_sided=fit["p_one_sided"], ci95=fit["ci95"], ci90=fit["ci90"], residual_sd=fit["residual_sd"],
               intercepts=fit["intercepts"], r5_category=cat, practically_meaningful=(cat == EXCEEDS))
    return out


def secondary_block(complete, values):
    """P6 / C3 secondary family with Holm."""
    rows = []
    tested = len(complete) >= MIN_COMPLETE
    for sid, kind, outcome, pair, label in SECONDARY:
        e = dict(id=sid, label=label, kind=kind, outcome=outcome, status=NOT_TESTED, estimate=None, se=None, df=None,
                 t=None, p_one_sided=None, p_holm=None, reject_holm=None, ci95=None, n=None)
        if tested:
            fit = paired_fit(complete, values, outcome, pair) if kind == "paired" else \
                slope_fit(complete, values, outcome)
            if fit["estimable"]:
                e.update(status=TESTED, estimate=fit["estimate"] if kind == "paired" else fit["b"], se=fit["se"],
                         df=fit["df"], t=fit["t"], p_one_sided=fit["p_one_sided"], ci95=fit["ci95"], n=fit["n"])
            else:
                e["status"] = f"Not estimable: {fit['reason']}"
        rows.append(e)
    if tested:
        adj = holm([e["p_one_sided"] if e["status"] == TESTED else 1.0 for e in rows])
        for e, pa in zip(rows, adj):
            if e["status"] == TESTED:
                e["p_holm"] = pa
                e["reject_holm"] = pa < ALPHA
    return rows


def by_arm_block(complete, values):
    """P7 per-arm descriptives over the complete replicates."""
    out = {}
    for N in ARMS:
        cell = dict(replicates=list(complete))
        for m in PER_RUN_METRICS:
            per = {str(r): values[(N, r)][m] for r in complete}
            d = describe(list(per.values()))
            cell[m] = dict(mean=d["mean"], ci95=d["ci95"], n=d["n"], values_by_replicate=per)
        out[str(N)] = cell
    return out


def _ordered_labels(values, canonical):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return [c for c in canonical if c in seen] + sorted((v for v in seen if v not in canonical), key=str)


def pooled_block(complete, test_maps_of, set_of, label_of, canonical):
    """P7 / C4: per arm, the id set pooled over the complete replicates by a manifest label (band or variant)."""
    id_idx = [i for i, s in enumerate(set_of) if s == "id"]
    labels = _ordered_labels([label_of[i] for i in id_idx], canonical)
    out = {}
    for N in ARMS:
        cells = {}
        for lab in labels:
            idx = [i for i in id_idx if label_of[i] == lab]
            if not complete:
                cells[str(lab)] = dict(F=None, O=None, Y=None, n_maps=len(idx), n_runs=0)
                continue
            F = O = 0
            for r in complete:
                maps = test_maps_of[(N, r)]
                for i in idx:
                    F += int(maps[i]["breaks"]) + int(maps[i]["tail_miss"])
                    O += int(maps[i]["max_combo_possible"])
            cells[str(lab)] = dict(F=F, O=O, Y=y_of(F, O), n_maps=len(idx), n_runs=len(complete))
        out[str(N)] = cells
    return out


def reference_block(led, refs, keys, set_of):
    """P7 / P8 / C8: the reference rows on the merged test pool, {name: {id, tempo, pattern, kind, note}}."""
    if not isinstance(refs, list):
        refuse("batch summary: 'references' must be a list")
    out = {}
    for i, ref in enumerate(refs):
        name = ref.get("name") if isinstance(ref, dict) else None
        if not isinstance(name, str) or not name:
            refuse(f"batch summary: reference {i} has no name")
        if name in out:
            refuse(f"batch summary: duplicate reference name {name!r}")
        if not ref.get("test_eval"):
            out[name] = dict({s: None for s in TEST_SETS}, kind=ref.get("kind"),
                             note="no test_eval in the batch summary")
            continue
        where = f"reference {name} test_eval"
        maps = eval_maps(read_checked(led, ref["test_eval"], ref.get("test_eval_sha256"), "reference_test_eval",
                                      where), where, allow_maps_json=True)
        check_keys(maps, keys, where, "the test manifest")
        out[name] = dict({s: set_outcome([m for m, t in zip(maps, set_of) if t == s]) for s in TEST_SETS},
                         kind=ref.get("kind"), note=None)
    return out


def _curve_y(F, O):
    try:
        return y_of(int(F), int(O))
    except (TypeError, ValueError):
        return None


def learning_curve(led, rec):
    """P7 / C5: one run's curves from its eval.jsonl -> {N, replicate, monitor, probe, note}; monitor = the rows named
    pd_monitor_ema, probe = the rows named pd_probe_ema, each row [decisions, Y, mean_acc, fc_rate]."""
    N, r = rec_key(rec)
    out = dict(N=N, replicate=r, **{key: [] for key, _ in CURVES}, note=None)
    key_of = {row_name: key for key, row_name in CURVES}
    p = rec.get("eval_jsonl")
    if not p:
        out["note"] = "no eval_jsonl in the batch summary; skipped"
        return out
    if not led.full(p).is_file():
        out["note"] = f"eval_jsonl not found; skipped: {p}"
        return out
    data, _ = led.read(p, "eval_jsonl", f"run {rec['name']} eval_jsonl")
    bad = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(row, dict) or row.get("name") not in key_of:
            continue
        summ = row.get("summary")
        summ = summ.get(MODE) if isinstance(summ, dict) else None
        summ = summ if isinstance(summ, dict) else {}
        out[key_of[row["name"]]].append([row.get("decisions"), _curve_y(summ.get("failures"), summ.get("objects")),
                                         summ.get("mean_acc"), summ.get("fc_rate")])
    if bad:
        out["note"] = f"{bad} unparsable line(s) skipped"
    return out


def settings_block(batch):
    return dict(arms=list(ARMS), replicates=list(REPLICATES), k={str(N): K_OF[N] for N in ARMS},
                k_definition="k = log8(N / 16)", delta=DELTA, alpha=ALPHA, min_complete_replicates=MIN_COMPLETE,
                zero_failure_rule=("Y = ln(F / O), F = sum of (breaks + tail_miss), O = sum of max_combo_possible over "
                                   "a set's maps; if F == 0, Y = ln(0.5 / O); O == 0 -> null"),
                test_sets=list(TEST_SETS),
                probe="every map of the run's probe eval (its first min(N, 256) practice maps)",
                eval_mode=MODE,
                primary_model=("OLS Y_id = alpha_r + b k + e, replicate fixed effects, classical standard error, "
                               "df = n - R - 1; H1: b < 0, one-sided p = P(T_df <= t)"),
                secondary_family=[s[0] for s in SECONDARY], holm_m=len(SECONDARY),
                secondary_direction="each one-sided in the predicted direction, estimate < 0",
                curve_names=list(CURVE_NAMES), curve_columns=list(CURVE_COLUMNS),
                budget_decisions_D=batch.get("D"), labels=list(LABELS))


def version_block():
    return dict(label=VERSION, script=HERE.name, script_sha256=sha256_file(HERE),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def primary_sentence(p, st):
    """P10: the primary in one sentence (estimate, 95% CI, one-sided p, R5 category)."""
    if p["status"] == NOT_TESTED:
        return (f"Primary (H1: b < 0): {NOT_TESTED}. With {p['n_replicates']} complete replicate set(s), fewer than "
                f"the {st['min_complete_replicates']} required, no estimate, interval or p-value is reported for the "
                "primary or the secondary tests; the descriptive results are reported alone.")
    if p["status"] != TESTED:
        return f"Primary (H1: b < 0): {p['status']}."
    s = (f"Primary (H1: b < 0): with replicate fixed effects, the OLS slope of Y_id (ln failures per combo object on "
         f"the in-distribution test set) on k = log8(N / 16) is b = {fnum(p['b'])} per 8x step of N (95% CI "
         f"{fci(p['ci95'])}; one-sided p = {fp(p['p_one_sided'])}; df = {p['df']}; {p['n']} runs in "
         f"{p['n_replicates']} complete replicate sets). R5 category (delta = {fnum(p['delta'], 2)}): "
         f"{p['r5_category']}.")
    if p["r5_category"] == EXCEEDS:
        s += (" Under the preregistered rule this slope is practically meaningful (its 90% CI lies entirely beyond the "
              "margin).")
    return s


def analyze(batch_path, root):
    """P1-P10 on a batch summary -> the result dict (not yet cleaned); refuses (SystemExit) on a P8 / C1 / C2 fault."""
    led = Ledger(root)
    batch = read_json(led, Path(batch_path).resolve(), "batch_summary", "batch summary")
    if not isinstance(batch, dict) or batch.get("schema") != SCHEMA_IN:
        refuse(f"batch summary: schema must be {SCHEMA_IN!r}")
    tm = batch.get("test_manifest")
    if not isinstance(tm, dict) or not tm.get("path"):
        refuse("batch summary: no test_manifest path")
    keys, set_of, band_of, variant_of = parse_test_manifest(
        read_checked(led, tm["path"], tm.get("sha256"), "test_manifest", "test manifest"))
    records = batch.get("runs")
    validate_runs(records)
    runs, entries, test_maps_of = [], {}, {}
    for rec in sorted(records, key=run_order):
        entry, maps = process_run(led, rec, keys, set_of)
        runs.append(entry)
        entries[(entry["N"], entry["replicate"])] = entry
        test_maps_of[(entry["N"], entry["replicate"])] = maps
    complete, excluded = complete_replicates(records)
    values = {key: run_values(e) for key, e in entries.items() if key[1] in complete}
    primary = primary_block(complete, values)
    secondary = secondary_block(complete, values)
    references = reference_block(led, batch.get("references") or [], keys, set_of)
    curves = {rec["name"]: learning_curve(led, rec) for rec in sorted(records, key=run_order)}
    descriptive = dict(by_arm=by_arm_block(complete, values),
                       by_band=pooled_block(complete, test_maps_of, set_of, band_of, BANDS),
                       by_variant=pooled_block(complete, test_maps_of, set_of, variant_of, VARIANTS),
                       references=references, learning_curves=curves)
    settings = settings_block(batch)
    wording = dict(primary_sentence=primary_sentence(clean(primary), settings))
    return dict(schema=SCHEMA_OUT, version=version_block(), inputs=led.manifest(), settings=settings,
                complete_replicates=complete, excluded_runs=excluded, runs=runs, primary=primary,
                secondary=secondary, descriptive=descriptive, power=power_table(), wording=wording)


# =============================================================================================================== text
def _set_text(name, d):
    if d is None:
        return f"{name} n/a"
    return (f"{name} F {fnum(d['F'])} O {fnum(d['O'])} fc_rate {fnum(d['fc_rate'])} mean_acc {fnum(d['mean_acc'])} "
            f"maps {fnum(d['n_maps'])}")


def _by_replicate(per):
    return ", ".join(f"r{k} {fnum(v)}" for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))) or "none"


def _ordered_items(cells, canonical):
    names = [str(c) for c in canonical]
    return [(k, cells[k]) for k in names if k in cells] + sorted((k, v) for k, v in cells.items() if k not in names)


def render_text(res):
    """Plain-text rendering of the result JSON (every number is a JSON value, formatted)."""
    v, st, p, d = res["version"], res["settings"], res["primary"], res["descriptive"]
    L = [f"STUDY PD ANALYSIS ({v['label']})", f"script {v['script']} sha256 {v['script_sha256']}",
         f"python {v['python']}, numpy {v['numpy']}, scipy {v['scipy']}; result schema {res['schema']}", "",
         "INPUTS (sha256, role, path)"]
    L += [f"  {f['sha256']}  {f['role']:<19} {f['path']}" for f in res["inputs"]]
    L += ["", "SETTINGS",
          "  arms N: " + ", ".join(f"{a} (k {st['k'][str(a)]})" for a in st["arms"]) + f"; {st['k_definition']}",
          "  replicates: " + ", ".join(str(r) for r in st["replicates"])
          + f"; at least {st['min_complete_replicates']} complete replicate sets are required for any test",
          f"  delta {fnum(st['delta'], 2)}; alpha {fnum(st['alpha'], 2)}; eval mode {st['eval_mode']}; budget D "
          f"{fnum(st['budget_decisions_D'])}",
          f"  outcome: {st['zero_failure_rule']}", f"  sets: {', '.join(st['test_sets'])}; probe: {st['probe']}",
          f"  primary model: {st['primary_model']}",
          f"  secondary family: {', '.join(st['secondary_family'])} (Holm, m = {st['holm_m']}; "
          f"{st['secondary_direction']})", ""]
    cr = res["complete_replicates"]
    L.append("COMPLETE REPLICATE SETS: " + (", ".join(str(r) for r in cr) if cr else "none"))
    L.append("EXCLUDED RUNS:" + ("" if res["excluded_runs"] else " none"))
    L += [f"  {e['name']}: {e['reason']}" for e in res["excluded_runs"]]
    L += ["", "PER RUN (Y = ln(F / O), lower is better; n/a = not evaluated)"]
    for r in res["runs"]:
        s = r["sets"]
        L.append(f"  {r['name']:<16} N {r['N']:>4}  r{r['replicate']}  k {r['k']}  complete {fnum(r['complete']):<5}  "
                 + "  ".join(f"Y_{x} {fnum(s[x]['Y'] if s[x] else None)}" for x in ("id", "tempo", "pattern", "probe"))
                 + f"  G {fnum(r['G'])}")
        L.append("      " + "; ".join(_set_text(x, s[x]) for x in ("id", "tempo", "pattern", "probe")))
    L += ["", "PRIMARY (H1: b < 0; OLS Y_id = alpha_r + b k + e, replicate fixed effects; one-sided)",
          f"  status: {p['status']}",
          f"  outcome {p['outcome']}; n {fnum(p['n'])} runs; complete replicate sets {fnum(p['n_replicates'])}; "
          f"df {fnum(p['df'])}",
          f"  b {fnum(p['b'])}; se {fnum(p['se'])}; t {fnum(p['t'], 3)}; one-sided p {fp(p['p_one_sided'])}",
          f"  95% CI {fci(p['ci95'])}; 90% CI {fci(p['ci90'])}; residual SD {fnum(p['residual_sd'])}",
          "  intercepts: " + (_by_replicate(p["intercepts"]) if p["intercepts"] else "n/a"),
          f"  R5 category (delta {fnum(p['delta'], 2)}): {p['r5_category'] or 'n/a'}; practically_meaningful: "
          f"{fnum(p['practically_meaningful'])}",
          "", f"SECONDARY FAMILY (Holm over {st['holm_m']} tests at alpha {fnum(st['alpha'], 2)}; "
              f"{st['secondary_direction']})"]
    for e in res["secondary"]:
        L.append(f"  {e['id']} [{e['kind']}, {e['outcome']}] {e['label']}")
        L.append(f"     status {e['status']}; estimate {fnum(e['estimate'])}; se {fnum(e['se'])}; df {fnum(e['df'])}; "
                 f"t {fnum(e['t'], 3)}; p {fp(e['p_one_sided'])}; p_holm {fp(e['p_holm'])}; reject_holm "
                 f"{fnum(e['reject_holm'])}; 95% CI {fci(e['ci95'])}; n {fnum(e['n'])}")
    L += ["", "DESCRIPTIVE BY ARM (complete replicate sets; mean, 95% t-interval, n, values by replicate)"]
    for a in st["arms"]:
        c = d["by_arm"][str(a)]
        L.append(f"  N {a} (replicates: {', '.join(str(r) for r in c['replicates']) or 'none'})")
        for m in PER_RUN_METRICS:
            x = c[m]
            L.append(f"    {m:<12} mean {fnum(x['mean'])}  95% CI {fci(x['ci95'])}  n {fnum(x['n'])}  | "
                     f"{_by_replicate(x['values_by_replicate'])}")
    for title, key, word, canon in (("ID SET BY BAND", "by_band", "band", BANDS),
                                    ("ID SET BY VARIANT", "by_variant", "variant", VARIANTS)):
        L += ["", f"{title} (pooled over the complete replicate sets: F, O, Y; maps per run, runs pooled)"]
        for a in st["arms"]:
            L.append(f"  N {a}: " + "; ".join(
                f"{word} {lab}: F {fnum(x['F'])} O {fnum(x['O'])} Y {fnum(x['Y'])} ({fnum(x['n_maps'])} maps/run, "
                f"{fnum(x['n_runs'])} runs)" for lab, x in _ordered_items(d[key][str(a)], canon)))
    L += ["", "REFERENCES (test sets)"]
    for name, x in d["references"].items():
        sets = " ".join(f"Y_{s} {fnum(x[s]['Y'] if x[s] else None)}" for s in st["test_sets"])
        L.append(f"  {name} ({x['kind']}): {sets}" + (f"  [note: {x['note']}]" if x["note"] else ""))
        L += [f"      {_set_text(s, x[s])}" for s in st["test_sets"] if x[s] is not None]
    lc, cols = d["learning_curves"], st["curve_columns"]
    L += ["", f"LEARNING CURVES (eval.jsonl rows: monitor = {st['curve_names'][0]}, probe = {st['curve_names'][1]}; "
              f"columns {', '.join(cols)})"]
    order = [r["name"] for r in res["runs"] if r["name"] in lc]
    order += sorted(n for n in lc if n not in order)
    for name in order:
        cur = lc[name]
        L.append(f"  {name} (N {fnum(cur['N'])}, r{fnum(cur['replicate'])})"
                 + (f"  [note: {cur['note']}]" if cur["note"] else ""))
        for key in ("monitor", "probe"):
            for row in cur[key]:
                L.append(f"    {key:<8} " + "  ".join(f"{col} {fnum(val)}" for col, val in zip(cols, row)))
    L += ["", "POWER TABLE (preregistered; P9 in the labelled definitions)",
          "     R    n   df  sigma  Sxx        se  ci95_half_width  ci90_half_width  min_detectable_one_sided"]
    for r in res["power"]:
        L.append(f"  {fnum(r['R']):>4} {fnum(r['n']):>4} {fnum(r['df']):>4} {fnum(r['sigma'], 2):>6} "
                 f"{fnum(r['sxx']):>4} {fnum(r['se']):>9} {fnum(r['ci95_half_width']):>16} "
                 f"{fnum(r['ci90_half_width']):>16} {fnum(r['min_detectable_one_sided']):>25}")
    L += ["", "PRIMARY SENTENCE", "  " + res["wording"]["primary_sentence"], "", "LABELLED DEFINITIONS AND CHOICES"]
    L += ["  " + x for x in st["labels"]]
    return "\n".join(L) + "\n"


def iter_strings(obj, skip=("inputs",)):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield str(k)
            yield from iter_strings(v, skip)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_strings(v, skip)
    elif isinstance(obj, str):
        yield obj


def wording_violations(res, text):
    """P10: words from FORBIDDEN anywhere; 'practically meaningful' only with the category 'Exceeds the margin'."""
    strings = list(iter_strings(res)) + [text]
    bad = []
    for s in strings:
        for pat in FORBIDDEN:
            if re.search(pat, s, flags=re.IGNORECASE):
                bad.append((pat, s[:120]))
    if (res.get("primary") or {}).get("r5_category") != EXCEEDS:
        for s in strings:
            if re.search(MEANINGFUL, s, flags=re.IGNORECASE):
                bad.append(("practically meaningful without the category Exceeds the margin", s[:120]))
    return bad


def run_analysis(batch, out_json, out_txt, root):
    """The CLI's work: analyze, render, check the wording, then create both outputs exclusively."""
    oj, ot = Path(out_json), Path(out_txt)
    if oj.resolve() == ot.resolve():
        refuse("--out-json and --out-txt name the same file")
    for p in (oj, ot):
        if p.exists():
            refuse(f"refusing to overwrite {p}")
        if not p.parent.is_dir():
            refuse(f"output folder does not exist: {p.parent}")
    res = clean(analyze(batch, root))
    data = json.dumps(res, sort_keys=True, indent=1, allow_nan=False) + "\n"
    res = json.loads(data)             # the text is rendered from the JSON values exactly as written
    text = render_text(res)
    bad = wording_violations(res, text)
    if bad:
        refuse(f"wording rule violated (P10): {bad[:3]}")
    with open(oj, "x", encoding="utf-8", newline="\n") as f:
        f.write(data)
    with open(ot, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return res, text


def _utf8_stdout():
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if enc not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Portable F1 analysis derivative (see the module docstring).")
    ap.add_argument("--batch", help="batch summary JSON (schema pd_batch_summary_v1)")
    ap.add_argument("--out-json", help="new PD_analysis.json (created exclusively)")
    ap.add_argument("--out-txt", help="new PD_analysis.txt (created exclusively)")
    ap.add_argument("--root", help="required input root for relative paths when analyzing a batch")
    ap.add_argument("--power-table", action="store_true", help="print the preregistered power table and exit")
    a = ap.parse_args(argv)
    _utf8_stdout()
    if a.power_table:
        sys.stdout.write(power_table_text(power_table()))
        return 0
    if not (a.batch and a.out_json and a.out_txt and a.root):
        ap.error("--batch, --root, --out-json and --out-txt are required (or --power-table)")
    _, text = run_analysis(a.batch, a.out_json, a.out_txt, root=a.root)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
