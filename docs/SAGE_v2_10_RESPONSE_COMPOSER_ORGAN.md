# SAGE v2.10 - Response Composer Organ

## Problem

v2.9 connected Memory Context Manager to Local Chat Loop.

However, the chat loop still often dumped internal analysis text:

```text
입력 내용을 상태로 변환했습니다.
추천 출력 전략: 먼저 state JSON을 만들고...
```

That is not proper sentence generation.

## Goal

v2.10 adds a Response Composer Organ.

Pipeline:

```text
user input
↓
CPU Language Core
↓
Memory Context Manager
↓
Response Composer Organ
↓
Korean sentence-plan response
```

## Important

This is not a neural LLM yet.

It is a symbolic / template / sentence-plan composer.

It generates answer sentences from:

```text
intent
concept card
state
selected memory context
safety boundary
```

## Example

```text
AGI가 뭔지 설명해줘
```

Expected behavior:

```text
AGI는 Artificial General Intelligence, 즉 인공일반지능을 뜻해...
현재 SAGE는 AGI라고 주장하는 단계가 아니라...
```

Instead of:

```text
추천 출력 전략: 먼저 state JSON...
```

## Safety

No new unsafe actions are enabled.

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
local sentence composition
read memory context
write chat logs
write memory candidates
```

## Run

```powershell
python tools\compose_response.py --text "AGI가 뭔지 설명해줘"
python tools\chat_with_sage.py --text "AGI가 뭔지 설명해줘"
python -m benchmarks.benchmark_v2_10_response_composer_organ
```
