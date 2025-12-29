#!/usr/bin/env bash

# Bash launcher for the RBM test using projection integrator.
# 
# - Reads the LAMMPS executable path from config/lammps_path.txt (if present, else uses "lmp")
# - Runs inputs/in.rbm_sphere_projection from the repo root
# - Writes the log to outputs/log.rbm_sphere_projection
# - Outputs angle time series to outputs/angles_sphere_projection.raw
# - Forwards any extra command-line arguments directly to LAMMPS
#
# Usage:
#  bash scripts/run_projection.sh
#  bash scripts/run_projection.sh -var N 10000 -var Dr 1.0 -var dt 0.3 -var nsteps 1000
# Requires bash; on Windows use Git Bash or WSL/WSL2

set -euo pipefail

# Determine script and repo paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to config file with executable
CONFIG_PATH="${REPO_ROOT}/config/lammps_path.txt"

if [[ -f "${CONFIG_PATH}" ]]; then
    LMP_EXE="$(tr -d '\r' < "${CONFIG_PATH}")"
else
    LMP_EXE="lmp"
fi

OUTDIR="${REPO_ROOT}/outputs"
INPUT="${REPO_ROOT}/inputs/in.rbm_sphere_projection"
LOG="${OUTDIR}/log.rbm_sphere_projection"

mkdir -p "${OUTDIR}"

# --- clean projection-related outputs, including log ---

# remove any file in outputs/ whose name contains "projection"
for f in "${OUTDIR}"/*projection*; do
    if [[ -f "$f" ]]; then
        rm -f "$f"
    fi
done

# --------------------------------------------------------------

echo "[run_projection.sh] Repo: ${REPO_ROOT}"
echo "[run_projection.sh] Executable: ${LMP_EXE}"
echo "[run_projection.sh] Input: ${INPUT}"
echo "[run_projection.sh] Log: ${LOG}"

cd "${REPO_ROOT}"

# Forward all arguments ($@) to LAMMPS:
exec "${LMP_EXE}" "$@" -log "${LOG}" -in "${INPUT}"
