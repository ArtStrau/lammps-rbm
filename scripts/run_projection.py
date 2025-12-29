#!/usr/bin/env python3
"""
Python launcher for the RBM test using projection integrator.

- Reads the LAMMPS executable path from config/lammps_path.txt (if present, else uses "lmp")
- Runs inputs/in.rbm_sphere_projection from the repo root
- Writes the log to outputs/log.rbm_sphere_projection
- Outputs angle time series to outputs/angles_sphere_projection.raw
- Forwards any extra command-line arguments directly to LAMMPS

Usage:
  python scripts/run_projection.py
  python scripts/run_projection.py -var N 10000 -var Dr 1.0 -var dt 0.3 -var nsteps 1000
Note: on some systems you may need to use "python3" instead of "python"
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

outdir = root / "outputs"
inp = "inputs/in.rbm_sphere_projection"
log_path = outdir / "log.rbm_sphere_projection"

# ensure outputs/ exists
outdir.mkdir(parents=True, exist_ok=True)

# --- clean projection-related outputs, including log ---
# delete all files in outputs/ whose name contains "projection"
for p in outdir.glob("*projection*"):
    if p.is_file():
        p.unlink()

# --------------------------------------------------------------

# all extra arguments after the script name are passed through to LAMMPS
extra_args = sys.argv[1:]

# Build command: your custom flags first, then log + input
cmd = [lmp, *extra_args, "-log", str(log_path), "-in", inp]

print("Running in:", root)
print("Command:", " ".join(cmd))

subprocess.run(cmd, cwd=root, check=True)
