# SAGE v2.0.6 - Cleanup & Retention Policy

## Goal

v2.0.6 adds a proposal-only cleanup and retention advisor.

As SAGE runs more continuous cycles, these folders can grow quickly:

```text
results/
configs/generated/
experiments/inbox/
memory/inbox/
logs/
```

The purpose of this version is not to delete or move files automatically.

The purpose is to let SAGE inspect repository growth and propose safe cleanup or retention actions for human review.

## Safety Rule

v2.0.6 is proposal-only.

It does not:

```text
delete files
move files
rename files
archive files automatically
approve memory automatically
commit to git
push to git
modify core source code
modify runtime behavior
```

## What It Produces

```text
results/v2_0_6_cleanup_retention_policy.json
experiments/inbox/v2_0_6_cleanup_retention_policy_proposal.json
```

The report contains:

```text
area stats
cleanup proposals
archive candidates
memory review candidates
log curation candidates
safety policy
```

## Run

```powershell
python tools/propose_cleanup_retention.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_6_cleanup_retention_policy
```

## Interpretation

A successful run means:

```text
SAGE can inspect repository growth.
SAGE can propose cleanup/retention actions.
SAGE does not execute cleanup automatically.
SAGE keeps human approval required.
```

## Next Direction

After v2.0.6, SAGE can move toward K-AI language experiments:

```text
v2.1 Korean Tokenization Probe
v2.2 Korean State-Based Understanding
v2.3 Dialectic Reflection Loop
```
