# SAGE v2.0.5 - Runtime Guard & Long-Run Monitor

## Summary

v2.0.5 adds a runtime guard for longer SAGE runs.

It monitors:

```text
cycles
failures
memory inbox growth
result file growth
experiment inbox growth
STOP file
```

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
python tools/run_guarded_continuous_runtime.py --max-cycles 10 --interval 1
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_5_runtime_guard_long_run_monitor
```

## Expected

```text
cycles_completed >= 1
failure_count: 0
memory_inbox_growth within limit
passed: True
```

## Commit

```powershell
git add sage_core/runtime_guard.py
git add sage_runtime/guarded_continuous_runtime.py
git add tools/run_guarded_continuous_runtime.py
git add benchmarks/benchmark_v2_0_5_runtime_guard_long_run_monitor.py
git add configs/guarded_continuous_runtime.json
git add docs/SAGE_v2_0_5_RUNTIME_GUARD_LONG_RUN_MONITOR.md
git add README_v2_0_5.md
git add results/v2_0_5_guarded_runtime
git add results/v2_0_5_guarded_runtime_smoke
git add results/v2_0_5_runtime_guard_long_run_monitor_smoke.json

git commit -m "Add SAGE v2.0.5 runtime guard and long-run monitor"
git tag v2.0.5
```
