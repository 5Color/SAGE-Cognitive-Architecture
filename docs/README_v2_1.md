# SAGE v2.1 - Memory Review Tool

## Summary

v2.1 adds a memory review tool.

It allows a human to:

```text
list memory/inbox proposals
show a proposal
approve a proposal
reject a proposal
write an audit record
```

It does not auto-approve memory.

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
python tools/review_memory.py list
python tools/review_memory.py report
```

Show a candidate:

```powershell
python tools/review_memory.py show --id <candidate_id>
```

Approve:

```powershell
python tools/review_memory.py approve --id <candidate_id> --reason "approved by human" --confirm
```

Reject:

```powershell
python tools/review_memory.py reject --id <candidate_id> --reason "rejected by human" --confirm
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_1_memory_review_tool
```

## Commit

Do not use `git add .`.

Use:

```powershell
git add sage_core/memory_review.py
git add sage_runtime/memory_review_runtime.py
git add tools/review_memory.py
git add benchmarks/benchmark_v2_1_memory_review_tool.py
git add configs/memory_review_tool.json
git add docs/SAGE_v2_1_MEMORY_REVIEW_TOOL.md
git add docs/versions/README_v2_1.md
git add results/v2_1_memory_review_tool_smoke.json
git add results/v2_1_memory_review_tool_smoke_result.json

git commit -m "Add SAGE v2.1 memory review tool"
git tag v2.1
```
