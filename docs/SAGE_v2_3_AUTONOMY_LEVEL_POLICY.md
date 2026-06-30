# SAGE v2.3 - Autonomy Level Policy

## Goal

v2.3 defines what SAGE may do automatically, what requires human approval, and what is forbidden.

This is needed before reducing human intervention.

## Core Principle

```text
SAGE may observe, think, propose, and write reports automatically.

SAGE may not approve memory, delete files, modify core code, commit to git,
push to git, access the network, or run arbitrary shell commands automatically.
```

## Levels

```text
Level 0: Manual
- Human manually runs every action.

Level 1: Assisted
- SAGE can inspect and suggest actions.

Level 2: Safe Auto
- SAGE can run whitelisted local loops and write proposals/results.

Level 3: Approval Required
- SAGE can prepare memory/archive decisions, but human approval is required.

Level 4: Restricted Self-Management
- SAGE can rank, summarize, deduplicate, and prioritize.
- Irreversible actions still require human approval.

Level 5: Full Autonomy Disabled
- Reserved boundary.
- Not enabled in this project.
```

## Current Default

```text
active_level: 2
```

Meaning:

```text
allowed:
- run_reflection_loop
- run_experiment_planner
- run_stability_probe
- run_cleanup_advisor
- write_results_json
- write_logs
- write_memory_proposal
- write_experiment_proposal
- create_stop_file

approval required:
- approve_memory
- reject_memory
- archive_results
- archive_generated_configs

forbidden:
- auto_approve_memory
- file_delete
- core_code_modification
- git_commit
- git_push
- network_access
- arbitrary_shell_execution
```

## Commands

Generate a report:

```powershell
python tools/check_autonomy_policy.py report
```

Check one action:

```powershell
python tools/check_autonomy_policy.py decide --action run_reflection_loop
python tools/check_autonomy_policy.py decide --action approve_memory
python tools/check_autonomy_policy.py decide --action file_delete
```

Smoke test:

```powershell
python -m benchmarks.benchmark_v2_3_autonomy_level_policy
```

## Meaning

A successful v2.3 means SAGE has an explicit autonomy boundary.

This does not make SAGE fully autonomous.

It makes limited autonomy safer and more understandable.
