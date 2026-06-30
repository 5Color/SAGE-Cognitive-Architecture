# SAGE v2.3 - Autonomy Level Policy

## Summary

v2.3 adds explicit autonomy levels.

Files:

```text
sage_core/autonomy_policy.py
sage_runtime/autonomy_policy_runtime.py
tools/check_autonomy_policy.py
benchmarks/benchmark_v2_3_autonomy_level_policy.py
configs/autonomy_level_policy.json
configs/autonomy_policy_runtime.json
docs/SAGE_v2_3_AUTONOMY_LEVEL_POLICY.md
docs/versions/README_v2_3.md
```

## Run

```powershell
python tools/check_autonomy_policy.py report
python tools/check_autonomy_policy.py decide --action run_reflection_loop
python tools/check_autonomy_policy.py decide --action approve_memory
python tools/check_autonomy_policy.py decide --action file_delete
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_3_autonomy_level_policy
```

## Commit

```powershell
git add sage_core/autonomy_policy.py
git add sage_runtime/autonomy_policy_runtime.py
git add tools/check_autonomy_policy.py
git add benchmarks/benchmark_v2_3_autonomy_level_policy.py
git add configs/autonomy_level_policy.json
git add configs/autonomy_policy_runtime.json
git add docs/SAGE_v2_3_AUTONOMY_LEVEL_POLICY.md
git add docs/versions/README_v2_3.md
git add results/v2_3_autonomy_level_policy_report.json
git add results/v2_3_autonomy_level_policy_smoke_result.json

git commit -m "Add SAGE v2.3 autonomy level policy"
git tag v2.3
```
