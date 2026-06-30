# SAGE v2.0.3 - Reflection Stability Probe

## Summary

v2.0.3 implements the experiment that SAGE proposed in v2.0.2:

```text
Stability Probe for Reflection Selection
```

It checks whether the exploratory policy still selects `curiosity_organ` when the policy weights are slightly perturbed.

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
python tools/run_reflection_stability_probe.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_3_reflection_stability_probe
```

## Expected Result

```text
variant_count: 6
selected_counts: {'curiosity_organ': 6}
target_selected_rate: 1.0
passed: True
```

## Commit

```powershell
git add sage_core/reflection_stability.py
git add sage_runtime/reflection_stability_runtime.py
git add tools/run_reflection_stability_probe.py
git add benchmarks/benchmark_v2_0_3_reflection_stability_probe.py
git add configs/reflection_stability_probe.json
git add configs/generated/stability_probe
git add docs/SAGE_v2_0_3_REFLECTION_STABILITY_PROBE.md
git add README_v2_0_3.md
git add results/v2_0_3_reflection_stability_probe.json
git add results/v2_0_3_reflection_stability_probe_smoke.json
git add results/v2_0_3_stability_probe

git commit -m "Add SAGE v2.0.3 reflection stability probe"
git tag v2.0.3
```
