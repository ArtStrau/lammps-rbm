#!/usr/bin/env python3
"""
Minimal launcher for the RBM projection example.

- Reads the LAMMPS executable path from config/lammps_path.txt (if present)
- Runs inputs/in.rbm_sphere_projection from the repo root
- Writes the log to outputs/log.lammps
- Forwards any extra command-line arguments directly to LAMMPS
"""

import sys
import subprocess
from pathlib import Path

# repo root = parent of scripts/
root = Path(__file__).resolve().parents[1]

# LAMMPS executable path: from config file if it exists, otherwise "lmp"
config_path = root / "config" / "lammps_path.txt"
if config_path.is_file():
    lmp = config_path.read_text().strip()
else:
    lmp = "lmp"

log = "outputs/log.lammps"
inp = "inputs/in.rbm_sphere_projection"

# ensure outputs/ exists
(root / "outputs").mkdir(parents=True, exist_ok=True)

# all extra arguments after the script name → passed through to LAMMPS
extra_args = sys.argv[1:]

# Build command: your custom flags first, then log + input
cmd = [lmp, *extra_args, "-log", log, "-in", inp]

print("Running in:", root)
print("Command:", " ".join(cmd))

subprocess.run(cmd, cwd=root, check=True)
