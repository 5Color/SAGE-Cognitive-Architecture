# SAGE v2.2 - Local Control Panel UI

## Summary

v2.2 adds a Streamlit-based local UI.

Files:

```text
sage_ui/__init__.py
sage_ui/control_panel.py
tools/run_sage_ui.py
Launch_SAGE_UI.cmd
Launch_SAGE_UI.ps1
requirements_ui.txt
docs/SAGE_v2_2_LOCAL_CONTROL_PANEL_UI.md
docs/versions/README_v2_2.md
```

## Run

```cmd
Launch_SAGE_UI.cmd
```

or:

```powershell
python tools/run_sage_ui.py
```

Open:

```text
http://127.0.0.1:7860
```

## Commit

Do not use `git add .`.

Use:

```powershell
git add sage_ui/__init__.py
git add sage_ui/control_panel.py
git add tools/run_sage_ui.py
git add Launch_SAGE_UI.cmd
git add Launch_SAGE_UI.ps1
git add requirements_ui.txt
git add docs/SAGE_v2_2_LOCAL_CONTROL_PANEL_UI.md
git add docs/versions/README_v2_2.md

git commit -m "Add SAGE v2.2 local control panel UI"
git tag v2.2
```
