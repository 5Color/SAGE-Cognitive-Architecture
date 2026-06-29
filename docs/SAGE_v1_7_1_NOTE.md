# SAGE v1.7.1 Planning Note

## Version Name

SAGE v1.7.1 - Adaptive Router Calibration + Organ Lifecycle Metrics

## Reason

v1.7 adaptive router worked, but Top2 usage was low.

Observed v1.7 pattern:

```text
top1: high
full: moderate
top2: low
```

## Goal

Improve routing calibration and add lifecycle diagnostics.

## Important Guardrail

No automatic deletion.
No automatic disabling.

SAGE may only recommend:

```text
keep_active
monitor
review_or_refactor_candidate
dormant_candidate
insufficient_data
```

## Next Step

If v1.7.1 succeeds, move to:

```text
v1.8 Organ Lifecycle Manager
```
