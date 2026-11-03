"""Portable procedural osu!standard simulation and visual RL components.

Relative user data paths resolve against OSU_SIM_ROOT, or the current working
directory when it is unset. Importing this package never searches a source
checkout, appends dependency directories, or changes the compiler environment.
"""
import os
from pathlib import Path

ROOT = str(Path(os.environ.get("OSU_SIM_ROOT", Path.cwd())).expanduser().resolve())
