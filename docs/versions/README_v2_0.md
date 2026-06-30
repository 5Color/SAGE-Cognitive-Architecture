# SAGE v2.0 - Emergent Reflection Loop

## Summary

SAGE v2.0 adds a small emergent-style reflection loop.

The system now lets multiple reflection organs propose different interpretations,
then uses an aggregator to select one reflection and store it as a memory proposal.

## Apply Patch

Use CMD to avoid PowerShell script policy issues:

```cmd
apply_patch.cmd C:\Users\sonsj\SAGE-v0
```

Alternative:

```powershell
python apply_patch.py --target C:\Users\sonsj\SAGE-v0
```

## Run

From SAGE repo root:

```powershell
python tools/run_emergent_reflection.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_emergent_reflection_smoke
```

## Expected Outputs

```text
results/v2_0_emergent_reflection.json
results/v2_0_emergent_reflection_smoke.json
results/v2_0_emergent_reflection_smoke_detail.json
logs/emergent_reflection.md
logs/v2_0_emergent_reflection_smoke.md
memory/inbox/*.json
```

## Commit

```powershell
git add sage_core/emergence.py
git add sage_runtime/emergent_reflection_loop.py
git add tools/run_emergent_reflection.py
git add benchmarks/benchmark_v2_0_emergent_reflection_smoke.py
git add configs/emergent_reflection.json
git add docs/SAGE_v2_0_EMERGENT_REFLECTION_LOOP.md
git add README_v2_0.md
git add results/v2_0_emergent_reflection.json
git add results/v2_0_emergent_reflection_smoke.json
git add results/v2_0_emergent_reflection_smoke_detail.json
git add logs/emergent_reflection.md
git add logs/v2_0_emergent_reflection_smoke.md
git add memory/inbox
git commit -m "Add SAGE v2.0 emergent reflection loop"
git tag v2.0
```
