#!/usr/bin/env python3
import pathlib
import subprocess
import sys

script_dir = pathlib.Path(__file__).resolve().parent
project_dir = script_dir.parent
command = [
    sys.executable,
    str(script_dir / "paper_harness.py"),
    "validate-completion",
    "--project-dir",
    str(project_dir),
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, check=False).returncode)
