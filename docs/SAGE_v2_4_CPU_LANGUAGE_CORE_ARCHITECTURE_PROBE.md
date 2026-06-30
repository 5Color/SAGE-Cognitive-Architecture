# SAGE v2.4 - CPU Language Core Architecture Probe

## Goal

v2.4 starts the CPU language-model direction.

This is not a full LLM.

It is a CPU-only language architecture probe for SAGE.

## Core Idea

Instead of building one huge model:

```text
input text
↓
giant transformer
↓
output text
```

SAGE uses a modular pipeline:

```text
input text
↓
Korean chunk tokenizer
↓
semantic state extractor
↓
approved memory retriever
↓
template composer
↓
short Korean response
```

## Why CPU-Friendly

CPU is not ideal for huge matrix multiplication.

CPU is good at:

```text
small modules
rule-based text processing
state extraction
memory lookup
JSON state handling
transparent control flow
small neural probes later
```

So v2.4 uses:

```text
large model replacement ❌
small language organ architecture ✅
```

## Components

```text
KoreanChunkTokenizer
- syllable tokens
- word tokens
- simple chunk tokens
- compression metrics

StateExtractor
- intent detection
- topic detection
- signal extraction

ApprovedMemoryRetriever
- read-only retrieval from memory/approved

TemplateComposer
- stable Korean output from state + memory hints

CPULanguageCore
- end-to-end pipeline
```

## Safety

```text
CPU only
no external model required
no network actions
memory read-only
no memory auto-approve
no file delete
no core code auto-modify
```

## Commands

Demo:

```powershell
python tools/run_cpu_language_core.py --demo
```

Custom input:

```powershell
python tools/run_cpu_language_core.py --text "SAGE 현재 진행상황 요약해줘"
```

Benchmark:

```powershell
python -m benchmarks.benchmark_v2_4_cpu_language_core
```

## Success Criteria

v2.4 passes if:

```text
input text is tokenized
chunks are generated
semantic state is extracted
approved memory retrieval is read-only
a Korean response is composed
benchmark passes on CPU without external model
```

## Meaning

A successful v2.4 does not prove that SAGE has a full language model.

It proves that SAGE can start using a CPU-friendly language organ pipeline:

```text
text → state → memory → composed response
```

This is the foundation for later small neural language modules.
