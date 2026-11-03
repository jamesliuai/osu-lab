# Environment and interface

The supported candidate check uses CPython 3.12.10 on Windows with NumPy 2.5.3, Numba 0.68.0, llvmlite 0.50.0, PyTorch 2.10.0+cu126, and pytest 9.1.1. This records the available validation environment; it is not a claim that CPU-only PyTorch wheels or other operating systems were tested. CPU validation hides CUDA and limits computation to one thread. The final verification report records what actually passed.

Core dependencies are pinned in `pyproject.toml`. Renderer, YAML configuration, and analysis dependencies are optional. No dependency binaries are redistributed. Offline installation with already available matching dependencies and setuptools is `python -m pip install --no-index --no-deps --no-build-isolation .`.

## Control and observation

Default control frequency: 60 Hz. Actions set an absolute cursor in normalized coordinates and a two-key held-state bitmask. The full renderer produces RGB tensors with shape [3, 192, 256]. The eight auxiliary values encode the previous cursor command (two), one-hot held keys (four), a first-step/continuation flag (zero on the first step and one afterward), and a reserved zero. Reset clears recurrent state. Source inspection establishes this interface; no new dataflow-invariance certification is claimed.

The simulator tracks circles, slider heads/ticks/repeats/tails, and spinners with NoFail episodes. Rate-aware simulation accepts per-map clock-rate tags. Rewards can include judgement scores, press cost, break/tail penalties, tracking and spinner shaping; the training recipe exposes all options. Some shaping and hold-floor losses use training-only state that is not an actor input.

## Paths

`OSU_SIM_ROOT` explicitly sets the base for relative pool paths; otherwise it is the process working directory. Pool selection is mandatory. The exported package never inserts the original checkout or private dependency directory into `sys.path` and never selects a local Songs folder.

The trainer checks `OSU_SIM_ROOT/sealed_registry.json` if it exists; `OSU_SEALED_REGISTRY` adds another registry. A missing registry provides no holdout protection. For new experiments supply a registry with entries containing paths and/or SHA-256 values, and inspect the trainer's startup checks. These controls protect only the supplied records; they do not establish that a study was preregistered.

The full renderer requires explicit `OSU_SIM_ASSET_DIR` and `OSU_SIM_FONT_AWESOME_DIR`; see the asset document. It is not needed by the CPU example or generated conformance tests.
