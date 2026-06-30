# SAGE v2.0.2 - Autonomous Experiment Planner

## Goal

v2.0.2 gives SAGE controlled autonomy.

SAGE can now inspect existing result JSON files and propose next experiments.

It does **not** execute those experiments automatically.

## Autonomy Level

```text
proposal_only
```

SAGE can:

```text
read approved result JSON files
summarize observations
generate next-experiment proposals
save proposals to experiments/inbox
recommend one proposal for human review
```

SAGE cannot:

```text
access the network
run shell commands by itself
delete files
disable organs
approve memory automatically
commit or push to git
```

## Main Files

```text
sage_core/experiment_planner.py
sage_runtime/autonomous_experiment_planner.py
tools/plan_next_experiment.py
benchmarks/benchmark_v2_0_2_autonomous_experiment_planner.py
configs/autonomous_experiment_planner.json
```

## Planner Logic

The planner checks:

```text
top_second_gap
selected_organ
policy benchmark result
memory inbox count
safety policy
```

Then it proposes experiments such as:

```text
Stability Probe for Reflection Selection
Policy Diversity Repair
Memory Inbox Review Tool
Experiment Planner Self-Check
```

## Why This Matters

This is the first step from passive reflection toward controlled self-directed research.

The system still requires human approval, but it can now suggest what to test next.

## Run

```powershell
python tools/plan_next_experiment.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_2_autonomous_experiment_planner
```

## Expected Output

```text
results/v2_0_2_autonomous_experiment_plan.json
results/v2_0_2_autonomous_experiment_planner_smoke.json
experiments/inbox/*.json
experiments/inbox/selected_next_experiment.json
```

## Success Criteria

```text
at least 2 proposals generated
selected proposal exists
selected proposal requires human approval
network/shell/delete/disable/memory-auto-approval/git actions disabled
proposal files written to experiments/inbox
passed: true
```
