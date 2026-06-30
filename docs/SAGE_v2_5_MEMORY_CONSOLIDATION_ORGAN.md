# SAGE v2.5 - Memory Consolidation Organ

## Goal

v2.5 adds bounded automatic memory consolidation.

This version moves beyond manual-only memory review.

The goal is to make SAGE more brain-like:

```text
candidate memory
↓
provisional memory
↓
validated memory
↓
approved long-term memory
```

## Why

A brain does not require manual approval for every internal memory update.

However, unrestricted automatic long-term memory approval can pollute the system.

So v2.5 uses bounded auto consolidation.

## Pipeline

```text
memory/inbox
↓
score candidate
↓
detect duplicate
↓
detect risk flags
↓
move to provisional / validated / approved / rejected
↓
write audit log
```

## Automatic Actions

Allowed:

```text
move low-confidence memory to memory/provisional
move stronger memory to memory/validated
strictly auto-approve low-risk result/policy memories
move duplicate candidates to memory/rejected
write audit log
```

Forbidden:

```text
file delete
network action
git action
arbitrary shell execution
core code auto-modification
risk-flagged auto approval
unbounded auto approval
```

## Strict Auto Approval

A memory can be moved directly to `memory/approved` only when:

```text
score >= min_auto_approve_score
risk_flags is empty
the memory contains result/benchmark evidence or policy/safety evidence
auto_approve_strict is true
```

Risk flags block auto approval.

Examples of risk flags:

```text
file_delete
network_access
git_push
arbitrary_shell
core_code_modification
full autonomy
overclaiming AGI/consciousness
```

## Commands

Run consolidation:

```powershell
python tools/consolidate_memory.py
```

JSON output:

```powershell
python tools/consolidate_memory.py --json
```

Smoke test:

```powershell
python -m benchmarks.benchmark_v2_5_memory_consolidation
```

## Audit Log

Every automatic move is logged to:

```text
memory/consolidation_log.jsonl
```

## Meaning

v2.5 does not mean SAGE has unrestricted memory autonomy.

It means SAGE can automatically consolidate low-risk memory while keeping a safety boundary.
