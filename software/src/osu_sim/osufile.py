"""Text-level .osu reader/writer (osu!standard). Geometry/timing math lives in beatmap.py.

The reader keeps raw values exactly as lazer's LegacyBeatmapDecoder would see
them (difficulty values are rounded to float32 later, where lazer uses float).
The writer emits format v14 files that the real host's native decoder accepts,
so generated curriculum maps can be played identically in osu-sim and lazer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingPoint:
    time: float
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int


@dataclass
class RawHitObject:
    x: float
    y: float
    time: float
    type: int
    hitsound: int
    # slider
    curve_type: str | None = None          # 'B' 'P' 'C' 'L'
    curve_points: list[tuple[float, float]] = field(default_factory=list)   # excludes the head
    slides: int = 1
    length: float | None = None
    # spinner
    end_time: float | None = None

    @property
    def kind(self) -> str:
        if self.type & 1:
            return "circle"
        if self.type & 2:
            return "slider"
        if self.type & 8:
            return "spinner"
        raise ValueError(f"unsupported hit object type {self.type}")

    @property
    def new_combo(self) -> bool:
        return bool(self.type & 4)

    @property
    def combo_skip(self) -> int:
        return (self.type >> 4) & 7


@dataclass
class OsuFile:
    version: int = 14
    general: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    difficulty: dict[str, float] = field(default_factory=dict)
    timing_points: list[TimingPoint] = field(default_factory=list)
    hit_objects: list[RawHitObject] = field(default_factory=list)
    breaks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mode(self) -> int:
        return int(self.general.get("Mode", "0").strip() or 0)


def _num(s: str) -> float:
    return float(s.strip())


def parse_osu(source: str | Path | bytes) -> OsuFile:
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith(".osu")):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.encode("utf-8")
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    out = OsuFile()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    head = lines[i].strip() if i < len(lines) else ""
    if head.startswith("osu file format v"):
        out.version = int(head[len("osu file format v"):].strip())
        i += 1
    section = None
    for line in lines[i:]:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1]
            continue
        if section in ("General", "Metadata", "Difficulty", "Editor"):
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip()
            if section == "General":
                out.general[k] = v
            elif section == "Metadata":
                out.metadata[k] = v
            elif section == "Difficulty":
                try:
                    out.difficulty[k] = float(v)
                except ValueError:
                    pass
        elif section == "Events":
            parts = s.split(",")
            if parts[0] in ("2", "Break") and len(parts) >= 3:
                out.breaks.append((_num(parts[1]), _num(parts[2])))
        elif section == "TimingPoints":
            p = s.split(",")
            if len(p) < 2:
                continue
            time, beat = _num(p[0]), _num(p[1])
            meter = int(_num(p[2])) if len(p) > 2 and p[2].strip() else 4
            sset = int(_num(p[3])) if len(p) > 3 and p[3].strip() else 0
            sidx = int(_num(p[4])) if len(p) > 4 and p[4].strip() else 0
            vol = int(_num(p[5])) if len(p) > 5 and p[5].strip() else 100
            unin = (p[6].strip() == "1") if len(p) > 6 and p[6].strip() else True
            eff = int(_num(p[7])) if len(p) > 7 and p[7].strip() else 0
            out.timing_points.append(TimingPoint(time, beat, meter, sset, sidx, vol, unin, eff))
        elif section == "HitObjects":
            p = s.split(",")
            if len(p) < 4:
                continue
            obj = RawHitObject(x=_num(p[0]), y=_num(p[1]), time=_num(p[2]), type=int(_num(p[3])),
                               hitsound=int(_num(p[4])) if len(p) > 4 and p[4].strip() else 0)
            if obj.type & 2:
                spec = p[5].split("|")
                obj.curve_type = spec[0].strip()[:1] or "B"
                pts = []
                for token in spec[1:]:
                    if ":" in token:
                        a, b = token.split(":")[:2]
                        pts.append((_num(a), _num(b)))
                obj.curve_points = pts
                obj.slides = int(_num(p[6])) if len(p) > 6 else 1
                obj.length = _num(p[7]) if len(p) > 7 and p[7].strip() else None
            elif obj.type & 8:
                obj.end_time = _num(p[5]) if len(p) > 5 else obj.time
            elif obj.type & 128:
                raise ValueError("osu!mania hold note in a standard map")
            out.hit_objects.append(obj)
    return out


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


def write_osu(osu: OsuFile, path: str | Path | None = None) -> str:
    """Emit a minimal v14 file (no audio, no events besides breaks)."""
    g = {"AudioFilename": "virtual", "AudioLeadIn": "0", "PreviewTime": "-1", "Countdown": "0",
         "SampleSet": "Normal", "StackLeniency": "0.7", "Mode": "0", "LetterboxInBreaks": "0",
         "WidescreenStoryboard": "0"}
    g.update(osu.general)
    md = {"Title": "osu-sim", "Artist": "osu-sim", "Creator": "osu-sim", "Version": "generated", "Source": "",
          "Tags": "", "BeatmapID": "0", "BeatmapSetID": "-1"}
    md.update(osu.metadata)
    d = {"HPDrainRate": 5, "CircleSize": 4, "OverallDifficulty": 8, "ApproachRate": 9, "SliderMultiplier": 1.4,
         "SliderTickRate": 1}
    d.update(osu.difficulty)
    lines = [f"osu file format v{osu.version}", "", "[General]"]
    lines += [f"{k}: {v}" for k, v in g.items()]
    lines += ["", "[Editor]", "DistanceSpacing: 1", "BeatDivisor: 4", "GridSize: 4", "TimelineZoom: 1", "",
              "[Metadata]"]
    lines += [f"{k}:{v}" for k, v in md.items()]
    lines += ["", "[Difficulty]"]
    lines += [f"{k}:{_fmt(v)}" for k, v in d.items()]
    lines += ["", "[Events]", "//Background and Video events", "//Break Periods"]
    lines += [f"2,{_fmt(a)},{_fmt(b)}" for a, b in osu.breaks]
    lines += ["//Storyboard Layer 0 (Background)", "", "[TimingPoints]"]
    for t in osu.timing_points:
        lines.append(",".join([_fmt(t.time), _fmt(t.beat_length), str(t.meter), str(t.sample_set),
                               str(t.sample_index), str(t.volume), "1" if t.uninherited else "0", str(t.effects)]))
    lines += ["", "", "[HitObjects]"]
    for o in osu.hit_objects:
        base = [_fmt(o.x), _fmt(o.y), _fmt(o.time), str(o.type), str(o.hitsound)]
        if o.type & 2:
            pts = "|".join(f"{_fmt(x)}:{_fmt(y)}" for x, y in o.curve_points)
            base += [f"{o.curve_type}|{pts}", str(o.slides), _fmt(o.length if o.length is not None else 0)]
            base += ["|".join(["0"] * (o.slides + 1)), "|".join(["0:0"] * (o.slides + 1)), "0:0:0:0:"]
        elif o.type & 8:
            base += [_fmt(o.end_time), "0:0:0:0:"]
        else:
            base += ["0:0:0:0:"]
        lines.append(",".join(base))
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
