#!/usr/bin/env python3
"""
Generic python launcher for the RBM test: sphere/circle (manifold) × geometric/projection (method)

- Reads the LAMMPS executable path from config/lammps_path.txt (if present, else uses "lmp")
- Runs inputs/in.rbm from the repo root (unless you pass -in ...)
- Writes a log file into outputs/ (unless you pass -log ...)
  Default: outputs/log_rbm_<manifold>_<method>.lammps  (if both -var manifold ... and -var method ... are provided)
  Fallback: outputs/log_rbm.lammps
- Deletes the default log file first if it already exists
- Outputs angle time series to outputs/angles_<manifold>_<method>.raw
- Forwards any extra command-line arguments directly to LAMMPS (e.g. -var N ... -var Dr ...)

Usage:
  python scripts/run_rbm.py -var manifold sphere -var method geometric  -var N 10000 -var Dr 1.0 -var dt 0.02 -var nsteps 1000
  python scripts/run_rbm.py -var manifold circle -var method projection -var N 10000 -var Dr 1.0 -var dt 0.02 -var nsteps 1000

Note: on some systems you may need to use "python3" instead of "python"
"""

import sys
import subprocess
from pathlib import Path

# --------------------------------------------------------------
# repo root = parent of scripts/
root = Path(__file__).resolve().parents[1]

# LAMMPS executable path: from config file if it exists, otherwise "lmp"
config_path = root / "config" / "lammps_path.txt"
if config_path.is_file():
    lmp = config_path.read_text(encoding="utf-8").strip()
    if not lmp:
        lmp = "lmp"
else:
    lmp = "lmp"

outdir = root / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# all extra arguments after the script name are passed through to LAMMPS
extra_args = sys.argv[1:]

# Parse last occurrences of: -var manifold <...>  and  -var method <...>
manifold = None
method = None
i = 0
while i + 2 < len(extra_args):
    if extra_args[i] == "-var":
        if extra_args[i + 1] == "manifold":
            manifold = extra_args[i + 2]
        elif extra_args[i + 1] == "method":
            method = extra_args[i + 2]
        i += 3
    else:
        i += 1

# --------------------------------------------------------------
# Default input file (unless user supplied -in ...)
inp = root / "inputs" / "in.rbm"
add_in = ("-in" not in extra_args)

# Default log file (unless user supplied -log ...)
add_log = ("-log" not in extra_args)
if add_log:
    if manifold and method:
        log_path = outdir / f"log_rbm_{manifold}_{method}.lammps"
    else:
        log_path = outdir / "log_rbm.lammps"

    # Delete prior default log to avoid mixing outputs
    if log_path.is_file():
        log_path.unlink()

# --------------------------------------------------------------
# Build command: pass-through flags first, then default log/input if needed
cmd = [lmp, *extra_args]
if add_log:
    cmd += ["-log", str(log_path)]
if add_in:
    cmd += ["-in", str(inp)]

print("Running in:", root)
print("Command:", " ".join(cmd))

subprocess.run(cmd, cwd=root, check=True)
