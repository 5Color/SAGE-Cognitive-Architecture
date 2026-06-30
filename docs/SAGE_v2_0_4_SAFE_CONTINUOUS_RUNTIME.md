# SAGE v2.0.4 - Safe Continuous Runtime

## Goal

v2.0.4 lets SAGE run repeatedly in a safe loop.

This is the first continuous runtime mode with limited autonomy.

## Autonomy Level

```text
Level 1: safe benchmark/runtime loop
```

SAGE can:

```text
run reflection loop
run autonomous experiment planner
run reflection stability probe
write results JSON
write logs
write experiment proposals to experiments/inbox
write generated configs under configs/generated
```

SAGE still cannot:

```text
access the network
run arbitrary shell commands
delete files
modify core code
disable/delete organs automatically
approve memory automatically
run git commit or git push
```

## Runtime Cycle

Each cycle can run:

```text
reflection
experiment planner
stability probe, every N cycles
result logging
safe sleep
STOP file check
```

## STOP File

The runtime checks:

```text
runtime_control/STOP
```

If this file exists, the runtime stops safely.

## Memory Safety

By default:

```text
create_memory_proposal: false
```

This prevents memory/inbox spam during continuous runtime.

If memory proposals are enabled later, `max_memory_inbox` stops the loop when the inbox grows too large.

## Run

```powershell
python tools/run_safe_continuous_runtime.py
```

For a short test:

```powershell
python tools/run_safe_continuous_runtime.py --max-cycles 2 --interval 0
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_4_safe_continuous_runtime
```

## Expected Outputs

```text
results/v2_0_4_continuous_runtime/summary.json
results/v2_0_4_safe_continuous_runtime_smoke.json
logs/v2_0_4_safe_continuous_runtime.md
```
