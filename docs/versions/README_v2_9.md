# SAGE v2.9 - Chat Context Integration

## Summary

v2.9 integrates v2.8 Memory Context Manager into v2.7 Local Chat Loop.

Files:

```text
sage_core/local_chat_loop.py
configs/local_chat_loop.json
benchmarks/benchmark_v2_9_chat_context_integration.py
docs/SAGE_v2_9_CHAT_CONTEXT_INTEGRATION.md
docs/versions/README_v2_9.md
```

## Run

```powershell
python tools/chat_with_sage.py --text "CPU Language Core와 safety policy 기준으로 SAGE 다음 단계 알려줘"
python tools/chat_with_sage.py
python -m benchmarks.benchmark_v2_9_chat_context_integration
```

## Commit

```powershell
git add sage_core/local_chat_loop.py
git add configs/local_chat_loop.json
git add benchmarks/benchmark_v2_9_chat_context_integration.py
git add docs/SAGE_v2_9_CHAT_CONTEXT_INTEGRATION.md
git add docs/versions/README_v2_9.md
git add results/v2_9_local_chat_context_result.json
git add results/v2_9_chat_context_integration_smoke_result.json

git commit -m "Integrate memory context into SAGE local chat loop"
git tag v2.9
```
