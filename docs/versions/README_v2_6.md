# SAGE v2.6 - Memory Summarizer

## Summary

v2.6 adds read-only memory summarization.

Files:

```text
sage_core/memory_summarizer.py
sage_runtime/memory_summarizer_runtime.py
tools/summarize_memory.py
benchmarks/benchmark_v2_6_memory_summarizer.py
configs/memory_summarizer.json
configs/memory_summarizer_runtime.json
docs/SAGE_v2_6_MEMORY_SUMMARIZER.md
docs/versions/README_v2_6.md
```

## Run

```powershell
python tools/summarize_memory.py
python tools/summarize_memory.py --json
python -m benchmarks.benchmark_v2_6_memory_summarizer
```

## Commit

```powershell
git add sage_core/memory_summarizer.py
git add sage_runtime/memory_summarizer_runtime.py
git add tools/summarize_memory.py
git add benchmarks/benchmark_v2_6_memory_summarizer.py
git add configs/memory_summarizer.json
git add configs/memory_summarizer_runtime.json
git add docs/SAGE_v2_6_MEMORY_SUMMARIZER.md
git add docs/versions/README_v2_6.md
git add memory/summaries/latest_memory_summary.md
git add memory/summaries/latest_memory_summary.json
git add results/v2_6_memory_summarizer_report.json
git add results/v2_6_memory_summarizer_smoke_result.json

git commit -m "Add SAGE v2.6 memory summarizer"
git tag v2.6
```
