# SAGE v2.1 - Memory Review Tool

## Goal

v2.1 adds a human-in-the-loop memory review tool.

SAGE can create memory proposals in:

```text
memory/inbox/
```

But these proposals should not automatically become long-term memory.

v2.1 makes it possible to inspect, approve, or reject memory proposals safely.

## Memory Lanes

```text
memory/inbox
- unapproved memory proposals

memory/approved
- human-approved memory

memory/rejected
- human-rejected memory

memory/review
- audit logs for approval/rejection decisions
```

## Safety Policy

v2.1 does not auto-approve memory.

```text
auto_approve_memory: false
auto_reject_memory: false
auto_delete_memory: false
file_delete: false
human confirmation required for approve/reject move
```

Approve/reject actions require an explicit command plus `--confirm`.

## Commands

List memory proposals:

```powershell
python tools/review_memory.py list
```

Show one proposal:

```powershell
python tools/review_memory.py show --id <candidate_id>
```

Approve one proposal:

```powershell
python tools/review_memory.py approve --id <candidate_id> --reason "useful memory" --confirm
```

Reject one proposal:

```powershell
python tools/review_memory.py reject --id <candidate_id> --reason "not useful or unsafe" --confirm
```

Show report:

```powershell
python tools/review_memory.py report
```

## Smoke Test

```powershell
python -m benchmarks.benchmark_v2_1_memory_review_tool
```

## Meaning

A successful run means:

```text
SAGE can list memory proposals.
SAGE can show a memory proposal.
Human-approved memory can move to memory/approved.
Human-rejected memory can move to memory/rejected.
Audit logs are saved in memory/review.
No memory is approved automatically.
```
