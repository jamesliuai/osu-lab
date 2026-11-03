# Reproduction and analysis

## Installed-copy conformance

Install the candidate into a clean environment. Copy `tests/` and `examples/` to a separate work directory outside the extracted source. Run from there with `OSU_SIM_EXPECT_UNDER` set to the environment's installed `site-packages`, `CUDA_VISIBLE_DEVICES=-1`, `NUMBA_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Use a new writable `NUMBA_CACHE_DIR` and disable automatic third-party pytest plugins with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```sh
python examples/generated_data_example.py --out-dir generated_demo
python -m pytest tests -p no:cacheprovider
```

All loaded `osu_sim` and `osu_rl` modules must originate under that installed path. Do not set `PYTHONPATH` to the private research repository. This release uses an `src/` layout so merely running in its top level does not expose an uninstalled package. The verification report distinguishes package tests from synthetic analysis checks and asset-dependent checks that were not run.

## Generated pools

`examples/recipe/build_recipe_pools.py` builds only procedural maps, uses one worker by default, refuses overlapping training/evaluation seed ranges, and refuses to overwrite existing pool files. Its new manifest records installed source hashes and portable pool basenames. Rebuilds should compare map text hashes and logical array contents; compressed containers can carry timestamps and need not have identical file hashes. Pool files contain object arrays and should only be loaded from trusted generated outputs.

The portable PD tool requires an explicit fresh output directory and exposes practice and monitor generation only. It does not expose the study's sealed-set creation command. These new outputs are illustrative reproductions of generator construction, not frozen study artifacts.

```sh
python tools/build_pd_pools.py train --replicate 1 --sizes 16,256,4096 --out-root new_practice_pools --workers 1
python tools/build_pd_pools.py monitor --out-root new_monitor_pool --workers 1
```

The practice builder constructs the full fixed sequence before selecting requested prefixes, so even a small prefix request performs the full sequence construction. The release check exercises its pure construction logic; it does not run a full study-sized pool build.

## Analysis

The portable F1 analysis implements the existing replicate-based analysis and retains its statistical kernels. It requires explicit input roots and refuses to overwrite analysis outputs. Its demonstration uses synthetic data; synthetic values must never be described as study results. The original frozen analysis and records remain unchanged. Accepted final aggregate evidence is now included under evidence/study; raw sealed inputs and checkpoints remain omitted. See STUDY_RESULTS.md. Public aggregate checks cannot recreate or independently authenticate the omitted raw observations.

```sh
python -m pip install scipy pytest
python tools/synthetic_analysis_example.py --out-root new_synthetic_demo
python -m pytest tools/test_pd_tools_synthetic.py -p no:cacheprovider
```

The standalone analysis tools were checked with CPython 3.13.5, NumPy 2.5.3, SciPy 1.18.1, and pytest 9.0.3. They do not import the simulation package, PyTorch, or study data. The simulation package itself remains pinned to Python 3.12.

## Integrity

`PUBLIC_ARTIFACT_MANIFEST.json` records the explicit payload, roles, hashes, provenance, and export transformations. `SHA256SUMS` covers that manifest and payload; it does not hash itself. Archive checksums are supplied next to the download. Verify an archive checksum before extraction, then verify the extracted payload. The public export does not contain Git metadata or private validation environments.
