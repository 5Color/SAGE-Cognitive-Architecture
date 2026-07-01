# SAGE v2.5 Fix Note - Memory Consolidation Thresholds

This patch is intended to be folded into the v2.5 commit before tagging.

## Fix

The first v2.5 smoke test could fail because:

```text
policy memories mentioning "file_delete is forbidden" were treated as risky
strong benchmark memories did not always reach strict auto-approval threshold
```

Updated behavior:

```text
dangerous terms in forbidden/blocked context do not count as risk flags
strong result/policy memories score high enough for strict auto approval
risk-flagged enabling language still blocks auto approval
```

## Run

```powershell
python -m benchmarks.benchmark_v2_5_memory_consolidation
```

Expected:

```text
passed: True
```
