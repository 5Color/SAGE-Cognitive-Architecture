# SAGE v2.0.6 - Cleanup & Retention Policy

## Summary

v2.0.6 adds a proposal-only cleanup advisor.

It scans:

```text
results/
configs/generated/
experiments/inbox/
memory/inbox/
logs/
docs/
```

It proposes cleanup or retention actions, but does not execute them.

## Apply Patch

```cmd
apply_patch.cmd C:\Users\sonsj\SAGE-v0
```

or:

```powershell
python apply_patch.py --target C:\Users\sonsj\SAGE-v0
```

## Run

```powershell
python tools/propose_cleanup_retention.py
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_0_6_cleanup_retention_policy
```

## Expected

```text
proposal_count >= 1
passed: True
```

## Commit

Do not use `git add .`.

Use:

```powershell
git add sage_core/retention_policy.py
git add sage_runtime/cleanup_retention_advisor.py
git add tools/propose_cleanup_retention.py
git add benchmarks/benchmark_v2_0_6_cleanup_retention_policy.py
git add configs/cleanup_retention_policy.json
git add docs/SAGE_v2_0_6_CLEANUP_RETENTION_POLICY.md
git add docs/versions/README_v2_0_6.md
git add results/v2_0_6_cleanup_retention_policy.json
git add results/v2_0_6_cleanup_retention_policy_smoke.json
git add results/v2_0_6_cleanup_retention_policy_smoke_result.json
git add experiments/inbox/v2_0_6_cleanup_retention_policy_proposal.json
git add experiments/inbox/v2_0_6_cleanup_retention_policy_smoke_proposal.json

git commit -m "Add SAGE v2.0.6 cleanup retention policy"
git tag v2.0.6
```
