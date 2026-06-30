# SAGE v2.0.5 - Runtime Guard & Long-Run Monitor

## Goal

v2.0.5 adds a guard layer on top of safe continuous runtime.

v2.0.4 proved that SAGE can run a short safe continuous loop.

v2.0.5 asks a different question:

```text
Can SAGE keep running while monitoring failures, file growth, memory inbox growth, and STOP conditions?
```

## What It Monitors

```text
cycle count
failure count
memory/inbox growth
results/ JSON growth
experiments/inbox growth
STOP file
allowed actions
forbidden actions
```

## Allowed Actions

```text
run_reflection_loop
run_autonomous_experiment_planner
run_reflection_stability_probe
write_results_json
write_logs
write_experiments_inbox
write_generated_configs
monitor_runtime_growth
stop_file_check
```

## Forbidden Actions

```text
network_access
arbitrary_shell_execution
file_delete
core_code_modification
organ_auto_disable
organ_auto_delete
memory_auto_approve
git_commit
git_push
```

## STOP File

To stop the runtime safely:

```powershell
New-Item -ItemType File runtime_control\STOP -Force
```

The runtime checks the STOP file between cycles.

## Run

```powershell
python tools/run_guarded_continuous_runtime.py --max-cycles 10 --interval 1
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_5_runtime_guard_long_run_monitor
```

## Expected Output

```text
results/v2_0_5_guarded_runtime/summary.json
results/v2_0_5_runtime_guard_long_run_monitor_smoke.json
```

## Meaning

This version does not make SAGE fully autonomous.

It makes SAGE safer to run for more cycles by adding monitoring and stop conditions.
