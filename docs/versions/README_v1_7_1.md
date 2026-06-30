# SAGE v1.7.1 - Adaptive Router Calibration + Organ Lifecycle Metrics

## Purpose

v1.7 succeeded, but Top2 usage was low.

v1.7.1 adds:

1. Calibrated adaptive router variants
2. Organ lifecycle diagnostics
3. Recommendation-only organ status
4. No automatic deletion
5. No automatic disabling

## New Files

```text
benchmarks/tasks/adaptive_compute_lifecycle_task.py

benchmarks/configs/
  anti_leak_adaptive_calibrated_balanced.json
  anti_leak_adaptive_calibrated_top2_friendly.json
  anti_leak_adaptive_calibrated_safe.json

benchmarks/benchmark_v1_7_1_lifecycle_calibration.py
```

## Run single config

```powershell
python run_experiment.py --config benchmarks/configs/anti_leak_adaptive_calibrated_balanced.json
```

## Run multi-seed

```powershell
python -m benchmarks.benchmark_v1_7_1_lifecycle_calibration --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-family 80
```

## Success Criteria

v1.7.1 is successful if:

1. calibrated router keeps accuracy close to v1.7 adaptive
2. Top2 usage increases compared with v1.7
3. compute saving remains meaningful
4. organ lifecycle output gives useful recommendations
5. no organ is automatically deleted or disabled

## Organ Lifecycle Policy

```text
auto_delete: false
auto_disable: false
recommend_only: true
human_approval_required: true
```

SAGE should not delete organs by default.
It should retire, archive, or reactivate organs based on evidence and human approval.
