# SAGE v2.4.1 - Config Loader Hotfix

## Summary

Fixes config loading for v2.4 CPU Language Core and v2.5 Memory Curator.

Problem:

```text
TypeError: __init__() got an unexpected keyword argument 'note'
```

Cause:

```text
configs/*.json included documentation-only fields such as "note",
but runtime dataclasses did not accept unknown keys.
```

Fix:

```text
Config loaders now ignore unknown/future fields.
```

## Run

```powershell
python tools/run_cpu_language_core.py --demo
python tools/run_cpu_language_core.py --text "SAGE 현재 진행상황 요약해줘"
python -m benchmarks.benchmark_v2_4_cpu_language_core
python tools/curate_memory.py
python -m benchmarks.benchmark_v2_5_memory_curator
```

## Commit

```powershell
git add sage_runtime/cpu_language_core_runtime.py
git add sage_runtime/memory_curator_runtime.py
git add docs/versions/README_v2_4_1.md

git commit -m "Fix config loader unknown fields"
git tag v2.4.1
```
