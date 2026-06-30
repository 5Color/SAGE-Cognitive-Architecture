# SAGE v2.2 - Local Control Panel UI

## Goal

v2.2 adds a local browser-based UI for SAGE.

The UI is designed to make SAGE easier to use without remembering every command.

## Launch

Recommended:

```cmd
Launch_SAGE_UI.cmd
```

PowerShell:

```powershell
.\Launch_SAGE_UI.ps1
```

Manual:

```powershell
python tools/run_sage_ui.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Features

```text
Dashboard
- memory counts
- result counts
- latest runtime summary
- STOP file status

Runtime
- run guarded runtime with max cycles
- create STOP file
- remove STOP file
- run cleanup advisor

Memory Review
- list memory/inbox candidates
- preview candidate
- approve candidate
- reject candidate
- require reason and typed confirmation

Results
- view recent result JSON files

Docs
- open SAGE docs inside UI

Safety
- view local safety policy
```

## Safety

The UI does not provide arbitrary shell execution.

It does not:

```text
delete files
auto-approve memory
modify core code
run git commit or push
perform external network actions
```

Approve/reject memory still requires explicit human action.

## Dependency

The UI uses Streamlit.

The launcher tries to install Streamlit if missing:

```powershell
python -m pip install streamlit
```

## Smoke Check

There is no separate benchmark for UI rendering.

Use:

```powershell
python tools/run_sage_ui.py
```

and verify that the browser opens successfully.
