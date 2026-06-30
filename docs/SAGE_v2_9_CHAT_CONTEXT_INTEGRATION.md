# SAGE v2.9 - Local Chat Loop + Memory Context Manager Integration

## Goal

v2.8 created a Memory Context Manager.

v2.9 connects it directly to the Local Chat Loop.

Old behavior:

```text
chat input
↓
CPU Language Core
↓
raw memory summary line
```

New behavior:

```text
chat input
↓
CPU Language Core
↓
Memory Context Manager
↓
query-specific context bundle
↓
response
↓
memory candidate proposal
```

## New command

Inside chat:

```text
/context <query>
```

Example:

```text
/context CPU Language Core와 safety policy 기준 다음 단계
```

## Safety

v2.9 remains local-only.

Forbidden:

```text
network actions
git actions
file delete
source memory delete
arbitrary shell actions
```

Allowed:

```text
read memory summary
read approved/validated/provisional memory
write derived context bundle
write chat logs
write memory candidate proposals
explicit /consolidate
explicit /summarize
```

## Run

```powershell
python tools/chat_with_sage.py --text "CPU Language Core와 safety policy 기준으로 SAGE 다음 단계 알려줘"
python tools/chat_with_sage.py
python -m benchmarks.benchmark_v2_9_chat_context_integration
```

## Meaning

SAGE no longer only says:

```text
Memory summary 참고:
# SAGE Memory Summary
```

It now selects relevant memory and exposes the selected context to the response.
