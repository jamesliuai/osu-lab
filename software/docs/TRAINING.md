# Training and evaluation

No training is run during release validation. The trainer currently selects CUDA. The CPU quick-start is a rules-and-generation demonstration, not a CPU training command.

## Portable generated recipe

From the extracted source release with the package installed:

```sh
python examples/recipe/build_recipe_pools.py --out-dir generated_pools --workers 1
python examples/recipe/make_portable_config.py --pools generated_pools --out recipe.json
python -m osu_sim.train --config recipe.json --out new_training_run
```

The config builder resolves both pools to absolute paths and verifies their presence. It reads only the included template; it never reads an earlier experiment or checkpoint. Defaults generate separate training and evaluation seeds. The supplied numeric recipe is a portable derivative of the generated-only recipe used as the procedural study's starting recipe. It is not a result, a tuned recommendation, or a complete reproduction of the final study. Source and export hashes are in the artifact manifest.

The last command requires a CUDA GPU, optional rendering dependencies, and separately cleared assets. Set `OSU_SIM_ASSET_DIR` and `OSU_SIM_FONT_AWESOME_DIR` to your permitted assets. No asset installer or unofficial font substitute is included. The approximate simple renderer is available for pipeline experiments, but changing to it changes the observation distribution and does not reproduce the calibrated-renderer study.

The template starts from scratch, has no resume/init/anchor checkpoint, and evaluates both current and averaged policy parameters on its explicit generated evaluation pool. PPO stores compact simulation states and renders observations on demand. Reward shaping, augmentation, architecture, rollout length, sampling, and moving-average settings remain explicit in the template. The hold-floor auxiliary loss uses training-only state. GPU training is not claimed bitwise deterministic.

`osu_sim.evaluate.SimEvaluator` supplies full-map evaluation to the trainer. Evaluation uses explicit pools and renderer settings; preserve these and the decoding mode when comparing runs. The scripted autopilot has access to map state and is a diagnostic reference, not a learned visual policy.

## Study reproduction status

Portable study pool and analysis tools are provided for inspection and synthetic checks. Final analysis inputs, reviewed findings, and any public study evidence are deferred until the study's existing final status, review evidence, report, results, analysis, provenance, and clean-data audit establish completion. This release does not execute the active queue, generate or inspect its sealed tests, or alter its environment. A later evidence supplement may extend this candidate after independent release checks.
