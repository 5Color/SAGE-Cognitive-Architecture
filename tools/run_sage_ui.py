from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ui_file = root / "sage_ui" / "control_panel.py"

    if not ui_file.exists():
        raise SystemExit(f"UI file not found: {ui_file}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_file),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "7860",
    ]

    print("Starting SAGE Local Control Panel UI...")
    print("Open: http://127.0.0.1:7860")
    subprocess.run(cmd, cwd=str(root), check=False)


if __name__ == "__main__":
    main()
