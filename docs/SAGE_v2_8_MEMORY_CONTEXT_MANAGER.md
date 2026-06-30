# SAGE v2.8 - Memory Context Manager

## Goal

v2.8 solves the next bottleneck after memory summarization.

The problem is not only memory size.

The real problem is context selection:

```text
현재 입력과 관련 있는 summary section / approved memory / validated memory만 고르는 것
```

## Pipeline

```text
user query
↓
query terms + topic inference
↓
memory summary section ranking
↓
approved / validated / provisional memory scoring
↓
compact context bundle
```

## Outputs

```text
results/v2_8_memory_context_bundle.json
```

The bundle contains:

```text
query
query_terms
inferred_topics
selected_summary_snippets
selected_memory_items
context_text
safety_policy
```

## Safety

v2.8 is read-only for memory.

It does not:

```text
delete memory
move memory
approve memory
reject memory
run shell
use network
run git
```

It only writes a derived context bundle.

## Commands

```powershell
python tools/build_memory_context.py --query "SAGE 현재 진행상황과 다음 단계 알려줘"
python tools/build_memory_context.py --query "CPU Language Core와 safety policy 기준으로 다음 단계 알려줘" --json
python -m benchmarks.benchmark_v2_8_memory_context_manager
```

## Meaning

SAGE no longer has to blindly read the entire memory summary.

It can construct a smaller, relevant context bundle for the current input.

This prepares v2.9, where Local Chat Loop can use the context bundle directly.
