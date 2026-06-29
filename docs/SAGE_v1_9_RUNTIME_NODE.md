# SAGE v1.9 - Runtime Node

## Goal

v1.9 turns SAGE from a benchmark-only project into a small persistent runtime.

It is not an autonomous internet agent.
It is not allowed to execute shell commands.
It does not delete or disable organs.

## Features

```text
persistent runtime state
organ registry read
memory inbox proposal
daily reflection log
safe idle mode
```

## Safe Runtime Policy

```text
network_actions: false
shell_actions: false
auto_delete_organs: false
auto_disable_organs: false
memory_approval_required: true
```

## Run

```powershell
python tools/run_sage_node.py --max-ticks 5 --tick-seconds 1
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v1_9_runtime_node_smoke
```

## Outputs

```text
runtime_state/state.json
memory/inbox/*.json
logs/daily_reflection.md
results/v1_9_runtime_node_smoke.json
```
