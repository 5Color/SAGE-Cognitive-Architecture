# SAGE v2.6 - Memory Summarizer

## Goal

v2.6 closes the memory lifecycle:

```text
generate → consolidate → summarize → retrieve
```

v2.5 allows bounded automatic memory consolidation.

That means memory can grow.

v2.6 creates compact derived summaries from:

```text
memory/approved
memory/validated
memory/provisional
```

## Safety

The summarizer is read-only for source memory.

It does not delete, move, approve, or reject source memory.

It only writes derived summaries:

```text
memory/summaries/latest_memory_summary.md
memory/summaries/latest_memory_summary.json
results/v2_6_memory_summarizer_report.json
```

## Commands

Run:

```powershell
python tools/summarize_memory.py
```

JSON output:

```powershell
python tools/summarize_memory.py --json
```

Smoke test:

```powershell
python -m benchmarks.benchmark_v2_6_memory_summarizer
```

## Meaning

SAGE can now keep long-term memory from expanding without compression.

This prepares the project for a Local Chat Loop, where the CPU Language Core can read compact memory summaries instead of scanning every memory file.
