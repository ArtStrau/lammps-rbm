#!/usr/bin/env bash
set -euo pipefail

# Determine script and repo paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to config file with executable
CONFIG_PATH="${REPO_ROOT}/config/lammps_path.txt"

if [[ -f "${CONFIG_PATH}" ]]; then
    LMP_EXE="$(<"${CONFIG_PATH}")"
else
    LMP_EXE="lmp"
fi

OUTDIR="${REPO_ROOT}/outputs"
INPUT="${REPO_ROOT}/inputs/in.rbm_sphere_projection"
LOG="${OUTDIR}/log.lammps"

mkdir -p "${OUTDIR}"

# --- clean projection-related outputs + generic log.lammps ---

# remove any file in outputs/ whose name contains "projection"
for f in "${OUTDIR}"/*projection*; do
    if [[ -f "$f" ]]; then
        rm -f "$f"
    fi
done

# remove generic log.lammps if present
if [[ -f "${LOG}" ]]; then
    rm -f "${LOG}"
fi

# --------------------------------------------------------------

echo "[run_projection.sh] Repo: ${REPO_ROOT}"
echo "[run_projection.sh] Executable: ${LMP_EXE}"
echo "[run_projection.sh] Input: ${INPUT}"
echo "[run_projection.sh] Log: ${LOG}"

cd "${REPO_ROOT}"

# Forward all arguments ($@) to LAMMPS:
exec "${LMP_EXE}" "$@" -log "${LOG}" -in "${INPUT}"
