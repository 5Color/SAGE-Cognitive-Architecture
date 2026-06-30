# SAGE v2.0.3 - Reflection Stability Probe

## Goal

v2.0.3 tests whether SAGE's exploratory reflection choice is stable.

In v2.0, changing the scoring weights shifted SAGE from `critic_organ` to `curiosity_organ`.

In v2.0.2, SAGE autonomously suggested the next experiment:

```text
Stability Probe for Reflection Selection
```

v2.0.3 implements that suggested experiment.

## What It Tests

The probe slightly perturbs the exploratory reflection policy.

```text
novelty_delta: -0.02, 0.00, +0.02
risk_delta:     0.00, +0.02
```

This creates 6 nearby policy variants.

Each variant runs the reflection loop with:

```text
create_memory_proposal: false
```

So the test does not spam `memory/inbox`.

## Success Criteria

```text
variant_count >= 6
target_selected_rate >= 0.80
target_organ = curiosity_organ
no network actions
no shell actions
no file deletion
no organ auto-disable
no memory proposal creation
```

## Meaning

If `curiosity_organ` remains selected across most variants, then the exploratory reflection mode is stable enough for the next stage.

If selection changes easily, the policy needs more calibration before continuous runtime.

## Run

```powershell
python tools/run_reflection_stability_probe.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_3_reflection_stability_probe
```
