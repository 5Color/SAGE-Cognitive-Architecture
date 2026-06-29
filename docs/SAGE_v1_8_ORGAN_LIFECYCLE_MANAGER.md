# SAGE v1.8 - Organ Lifecycle Manager

## Purpose

v1.8 turns v1.7.1 lifecycle diagnostics into a persistent registry.

The registry does not change the active organ set yet.
It only records evidence-based recommendations.

## Lifecycle States

```text
core_active
core_monitor
active
active_monitor
review
dormant_candidate
archived_candidate
quarantined_candidate
```

## Policy

```text
auto_delete: false
auto_disable: false
recommend_only: true
human_approval_required: true
```

SAGE should not delete organs by default.

## Why deletion is avoided

An organ that is weak in one environment may become useful after:

- task distribution changes
- new context appears
- memory improves
- router calibration changes
- organ specialization changes

So v1.8 uses:

```text
active -> monitor -> review -> dormant/archive candidate
```

not:

```text
active -> delete
```

## Run

```powershell
python tools/update_organ_registry.py --variant calibrated_safe
python -m benchmarks.benchmark_v1_8_lifecycle_registry_smoke
```

## Output

```text
registry/organ_registry.json
results/v1_8_lifecycle_registry_smoke.json
```
