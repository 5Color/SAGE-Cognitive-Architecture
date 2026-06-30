# SAGE Experiment Log Index

## Logs

* `docs/logs/EXPERIMENT_LOG_v0_to_v1_5.md`

  * Initial prototype through anti-leak evidence routing.

* `docs/logs/EXPERIMENT_LOG_v1_6.md`

  * Core refactor, config runner, anti-leak task plugin, and sparse evidence router.

## Current Focus

SAGE is currently focused on state-based adaptive organ routing, runtime persistence, organ lifecycle management, memory inbox proposals, and emergent reflection loops.

The current short-term goal is not to claim AGI, but to build and test the minimum conditions for an adaptive organ ecosystem:

```text
state
memory
router
organ lifecycle
reflection
selection pressure
human-approved memory
```

## Guardrail

SAGE is an experimental architecture research project.

Current results are synthetic benchmark and runtime prototype evidence, not proof of AGI or true emergence.

SAGE should not confuse logs, random variation, or interesting-looking behavior with real emergence. A behavior should only be considered more valuable if it improves later decisions, can be reused, or transfers to a new situation.

---

# v2.0 - Emergent Reflection Loop Calibration

## Goal

Test whether changing the reflection scoring weights can change SAGE's selected reflection organ.

This experiment checks whether SAGE's reflection policy is controllable through explicit scoring weights.

## Background

Before calibration, SAGE selected the `critic_organ`.

The Critic Organ's role is to prevent SAGE from confusing random variation or log generation with true emergence.

After calibration, the scoring policy was shifted toward exploration by increasing the weight of `novelty`.

## Change

The aggregator scoring function was changed to increase novelty weight while still keeping confidence, reuse value, risk, cost, and contradiction penalties.

```text
score =
  0.25 * confidence
+ 0.29 * novelty
+ 0.20 * reuse_value
+ 0.14 * agreement_bonus
- 0.08 * cost
- 0.08 * risk
- contradiction_penalty
```

## Design Intention

The goal of this calibration is to make SAGE slightly more exploratory.

This does not mean SAGE should blindly prefer strange or random outputs.

The scoring still includes:

```text
confidence
reuse_value
agreement_bonus
cost penalty
risk penalty
contradiction penalty
```

The intended behavior is:

```text
critic-biased mode:
- safer
- more conservative
- better for guardrails

curiosity-biased mode:
- more exploratory
- better for discovering new reflection patterns
- still controlled by risk, cost, and contradiction penalties
```

The current v2.0 policy intentionally uses the curiosity-biased mode because the purpose of this stage is to explore emergent reflection candidates.

## Result

```text
selected_organ: curiosity_organ
candidate_count: 5.0
organ_diversity: 1.0
mean_candidate_score: 0.47794
top_score: 0.5378
top_second_gap: 0.0163
mean_novelty: 0.514
mean_reuse_value: 0.77
memory_inbox: 7 -> 8
passed: True
```

## Interpretation

The selected organ changed from `critic_organ` to `curiosity_organ`.

This shows that SAGE's reflection policy is controllable through explicit scoring weights.

Increasing novelty makes the system more exploratory, while risk, cost, and contradiction penalties still prevent uncontrolled selection.

This is not proof of AGI or true emergence.

It is evidence that SAGE can expose and tune its internal reflection preference.

## Important Note

The `top_second_gap` is small.

```text
top_second_gap: 0.0163
```

This means the Curiosity Organ did not win by a large margin.

The correct interpretation is not:

```text
SAGE has become fully curiosity-driven.
```

The correct interpretation is:

```text
The scoring change shifted SAGE's reflection preference from critic-biased behavior toward curiosity-biased behavior.
```

## Current Conclusion

v2.0 successfully demonstrates a minimal emergent reflection loop:

```text
multiple organs generate candidates
aggregator scores candidates
one reflection is selected
memory proposal is created
smoke test passes
```

The next step should be to make the scoring policy configurable instead of hard-coded, so SAGE can switch between modes such as:

```text
conservative
exploratory
memory-focused
balanced
```

## Commit

```text
Calibrate SAGE v2.0 exploratory reflection weights
```
