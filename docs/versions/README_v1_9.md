# SAGE v1.9 - Runtime Node

## Summary

SAGE v1.9 adds a small persistent runtime node.

This version is designed for laptop-first development.

## Apply Patch

Recommended because PowerShell `.ps1` may be blocked:

```cmd
apply_patch.cmd C:\Users\sonsj\SAGE-v0
```

Alternative:

```powershell
python apply_patch.py --target C:\Users\sonsj\SAGE-v0
```

## Run

From the SAGE repo root:

```powershell
python tools/run_sage_node.py --max-ticks 5 --tick-seconds 1
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v1_9_runtime_node_smoke
```

## Commit

```powershell
git add sage_core/runtime_state.py
git add sage_core/memory_store.py
git add sage_runtime/sage_node.py
git add tools/run_sage_node.py
git add benchmarks/benchmark_v1_9_runtime_node_smoke.py
git add configs/runtime_node.json
git add docs/SAGE_v1_9_RUNTIME_NODE.md
git add README_v1_9.md
git add runtime_state/state.json
git add runtime_state/smoke_state.json
git add logs/daily_reflection.md
git add logs/v1_9_runtime_smoke_reflection.md
git add memory/inbox
git add results/v1_9_runtime_node_smoke.json
git commit -m "Add SAGE v1.9 runtime node"
git tag v1.9
```
