# SAGE v1.8 - Organ Lifecycle Manager

## Goal

Create a persistent organ registry from v1.7.1 lifecycle diagnostics.

## Important

This version does **not** delete organs.
This version does **not** disable organs.
It only recommends lifecycle status.

## Apply Patch

From the extracted patch folder:

```powershell
.\apply_patch.ps1 -Target "C:\Users\sonsj\SAGE-v0"
```

Or manually copy the folders into the SAGE repo root.

## Build Registry

From SAGE repo root:

```powershell
python tools/update_organ_registry.py --variant calibrated_safe
```

Alternative:

```powershell
python tools/update_organ_registry.py --variant calibrated_balanced
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v1_8_lifecycle_registry_smoke
```

## Commit

```powershell
git add sage_core/lifecycle.py
git add tools/update_organ_registry.py
git add benchmarks/benchmark_v1_8_lifecycle_registry_smoke.py
git add registry/organ_registry.example.json
git add registry/organ_registry.json
git add docs/SAGE_v1_8_ORGAN_LIFECYCLE_MANAGER.md
git add README_v1_8.md
git add results/v1_8_lifecycle_registry_smoke.json
git commit -m "Add SAGE v1.8 organ lifecycle manager"
git tag v1.8
```
