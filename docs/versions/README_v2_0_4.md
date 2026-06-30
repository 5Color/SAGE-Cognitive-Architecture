# SAGE v2.0.4 - Safe Continuous Runtime

## Summary

v2.0.4 adds a safe continuous runtime loop.

It repeatedly runs allowed SAGE modules:

```text
reflection loop
experiment planner
reflection stability probe
```

It does not allow arbitrary shell commands, network access, file deletion, git actions, core code modification, organ auto-disable/delete, or memory auto-approval.

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
python tools/run_safe_continuous_runtime.py
```

Short run:

```powershell
python tools/run_safe_continuous_runtime.py --max-cycles 2 --interval 0
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_4_safe_continuous_runtime
```

## STOP

Create this file to stop the runtime:

```text
runtime_control/STOP
```

## Commit

```powershell
git add sage_runtime/safe_continuous_runtime.py
git add tools/run_safe_continuous_runtime.py
git add benchmarks/benchmark_v2_0_4_safe_continuous_runtime.py
git add configs/safe_continuous_runtime.json
git add docs/SAGE_v2_0_4_SAFE_CONTINUOUS_RUNTIME.md
git add README_v2_0_4.md
git add results/v2_0_4_safe_continuous_runtime_smoke.json
git add results/v2_0_4_continuous_runtime_smoke

git commit -m "Add SAGE v2.0.4 safe continuous runtime"
git tag v2.0.4
```
