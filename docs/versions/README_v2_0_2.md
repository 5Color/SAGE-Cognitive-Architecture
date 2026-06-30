# SAGE v2.0.2 - Autonomous Experiment Planner

## Summary

v2.0.2 adds a controlled autonomous experiment planner.

SAGE can now read existing results and propose next experiments.

It does not execute them automatically.

## Apply Patch

Use CMD:

```cmd
apply_patch.cmd C:\Users\sonsj\SAGE-v0
```

Alternative:

```powershell
python apply_patch.py --target C:\Users\sonsj\SAGE-v0
```

## Run

```powershell
python tools/plan_next_experiment.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_2_autonomous_experiment_planner
```

## Expected Result

```text
proposals >= 2
selected proposal exists
selected proposal requires human approval
passed: True
```

## Commit

```powershell
git add sage_core/experiment_planner.py
git add sage_runtime/autonomous_experiment_planner.py
git add tools/plan_next_experiment.py
git add benchmarks/benchmark_v2_0_2_autonomous_experiment_planner.py
git add configs/autonomous_experiment_planner.json
git add docs/SAGE_v2_0_2_AUTONOMOUS_EXPERIMENT_PLANNER.md
git add README_v2_0_2.md
git add results/v2_0_2_autonomous_experiment_plan.json
git add results/v2_0_2_autonomous_experiment_planner_smoke.json
git add experiments/inbox

git commit -m "Add SAGE v2.0.2 autonomous experiment planner"
git tag v2.0.2
```
