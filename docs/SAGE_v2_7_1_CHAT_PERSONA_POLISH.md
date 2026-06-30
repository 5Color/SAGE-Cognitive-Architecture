# SAGE v2.7.1 - Chat Persona Polish

## Goal

v2.7 local chat loop works, but early conversational turns are too mechanical.

v2.7.1 adds lightweight persona handling for:

```text
greeting
identity
capabilities
help
```

## Examples

```text
안녕?
→ 안녕. 나는 SAGE의 로컬 대화 루프야.

넌 누구야
→ 나는 SAGE(Self-organizing Adaptive Generative Ecosystem)의 local chat loop야.

뭐 할 수 있어?
→ status, summary, consolidate, summarize commands...
```

## Safety

This is only response polish.

No new external actions are allowed.

```text
network actions: false
git actions: false
file delete: false
arbitrary shell actions: false
```

## Run

```powershell
python tools/chat_with_sage.py
python -m benchmarks.benchmark_v2_7_1_chat_persona_polish
```
