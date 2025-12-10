#!/usr/bin/env python3
"""
compute_pdf_from_angles.py

Postprocess a LAMMPS dump of per-atom angles and produce

  1) a cleaned angle sample file (one angle per line), and
  2) a normalized probability density function (PDF) over [0, pi].

The script is method independent: it only sees angles. Specify which
method they came from (projection / geometric / ...) so that the
output filenames are tagged accordingly.

Typical usage from the repo root:

    # projection data
    python scripts/compute_pdf_from_angles.py \
        --input outputs/angles_sphere_projection.raw \
        --method projection \
        --N 1000 --Dt 1.0 --Dr 3.0 --dt 0.1 --nsteps 10000 [--nbins 180 --no-add-zero]

    # geometric data
    python scripts/compute_pdf_from_angles.py \
        --input outputs/angles_sphere_geometric.raw \
        --method geometric \
        --N 1000 --Dt 1.0 --Dr 3.0 --dt 0.1 --nsteps 10000 [--nbins 180 --no-add-zero]

Note: by default, the script prepends a pdf(0)=0 point to the PDF output as follows from theory;
Add the option --no-add-zero not to do this, using the lammps data only

This will write, by default, files like

    data/angles_lammps_proj_Dr3.0_dt0.01.dat
    data/pdf_lammps_proj_Dr3.0_dt0.01.dat

using whitespace-separated columns and '#' comment headers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np


def load_angles_from_dump(path: Path) -> np.ndarray:
    """
    Parse a LAMMPS custom dump of the form

        ITEM: TIMESTEP
        <t>
        ITEM: NUMBER OF ATOMS
        <N>
        ITEM: BOX BOUNDS ...
        <3 lines>
        ITEM: ATOMS f_ang_out
        <N lines, one angle per line>
        (repeats...)

    and return all angles (in radians) as a flat numpy array.
    """
    angles: list[float] = []

    with path.open("r") as f:
        N: int | None = None
        while True:
            line = f.readline()
            if not line:
                break  # EOF

            line = line.strip()
            if not line:
                continue

            if line.startswith("ITEM: NUMBER OF ATOMS"):
                # next line is the number of atoms
                n_line = f.readline()
                if not n_line:
                    raise EOFError("Unexpected end of file after 'ITEM: NUMBER OF ATOMS'.")
                N = int(n_line.strip())

            elif line.startswith("ITEM: BOX BOUNDS"):
                # skip three bounds lines
                for _ in range(3):
                    if not f.readline():
                        raise EOFError("Unexpected end of file while skipping BOX BOUNDS.")

            elif line.startswith("ITEM: ATOMS"):
                if N is None:
                    raise ValueError("NUMBER OF ATOMS not read before ATOMS block.")
                for _ in range(N):
                    val_line = f.readline()
                    if not val_line:
                        raise EOFError("Unexpected end of file while reading angles.")
                    # one number per line (angle in radians) – use first column
                    first = val_line.strip().split()[0]
                    angles.append(float(first))

            # all other ITEM: lines (e.g. TIMESTEP) are ignored

    return np.asarray(angles, dtype=float)


def format_param(value: float) -> str:
    """
    Format a float nicely for filenames.

    Current convention (default):
        3.0   -> "3.0"
        0.01  -> "0.01"

    If you prefer an alternative (shorter) convention instead:
        3.0   -> "3"
        0.01  -> "0.01"
    then replace the body below with:  return f"{value:.8g}"
    """

    # --- current behavior: keep one decimal for integer-like values ---
    if float(value).is_integer():
        return f"{value:.1f}"      # 3, 3.0 or 3.00 -> "3.0"
    # compact representation for non-integer values
    return f"{value:.8g}"          # 0.01 -> "0.01"
    # ------------------------------------------------------------------

    # Alternative behavior (shorter; uncomment and remove the code above):
    # return f"{value:.8g}"        # compact representation applies to all values


def method_tag(method: str) -> str:
    """Map a verbose method name to a short tag for filenames."""
    m = method.lower()
    if m.startswith("proj"):
        return "proj"
    if m.startswith("geom"):
        return "geom"
    return m  # fallback: use as-is


def compute_pdf(
    angles: np.ndarray,
    nbins: int,
    add_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized PDF over [0, pi] from a 1D array of angles.

    Returns (theta, pdf) where theta are bin centers in [0, pi],
    and pdf(theta) is normalized so that the integral over [0, pi] is ~1.

    If add_zero is True, a point (theta=0, pdf=0) is prepended.
    This encodes the theoretical behavior at 0, even though
    the histogram has no bin exactly at 0.
    """
    if angles.size == 0:
        raise ValueError("No angles found in input file.")

    counts, edges = np.histogram(
        angles,
        bins=nbins,
        range=(0.0, math.pi),
        density=True,  # normalize to a PDF
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    if add_zero:
        centers = np.concatenate(([0.0], centers))
        counts = np.concatenate(([0.0], counts))

    return centers, counts


def write_angles_file(
    path: Path,
    angles: np.ndarray,
    *,
    method: str,
    N: int | None,
    Dt: float | None,
    Dr: float,
    dt: float,
    nsteps: int | None,
) -> None:
    """Write one angle per line with a small header."""
    with path.open("w") as f:
        f.write(f"# method: {method}\n")
        f.write(
            "# N={N} Dt={Dt} Dr={Dr} dt={dt} nsteps={nsteps}\n".format(
                N=N if N is not None else "NA",
                Dt=Dt if Dt is not None else "NA",
                Dr=Dr,
                dt=dt,
                nsteps=nsteps if nsteps is not None else "NA",
            )
        )
        f.write("# column: angle (rad)\n")
        for theta in angles:
            f.write(f"{theta:.9f}\n")


def write_pdf_file(
    path: Path,
    theta: np.ndarray,
    pdf: np.ndarray,
    *,
    method: str,
    N: int | None,
    Dt: float | None,
    Dr: float,
    dt: float,
    nsteps: int | None,
    nbins: int | None,
) -> None:
    """Write angle–PDF pairs with a small header."""
    with path.open("w") as f:
        f.write(f"# method: {method} (LAMMPS numerics)\n")
        f.write(
            "# N={N} Dt={Dt} Dr={Dr} dt={dt} nsteps={nsteps} nbins={nbins}\n".format(
                N=N if N is not None else "NA",
                Dt=Dt if Dt is not None else "NA",
                Dr=Dr,
                dt=dt,
                nsteps=nsteps if nsteps is not None else "NA",
                nbins=nbins if nbins is not None else "NA",
            )
        )
        f.write("# columns: angle(rad)  pdf(angle)\n")
        for t, p in zip(theta, pdf):
            f.write(f"{t:.9f} {p:.9e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute angle PDF from LAMMPS dump of per-atom angles."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="outputs/angles_sphere_projection.raw",
        help="Path to LAMMPS dump with angles (relative to repo root).",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Method label, e.g. 'projection' or 'geometric'. Used for filenames and headers.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Time step dt (used in headers and filenames).",
    )
    parser.add_argument(
        "--Dr",
        type=float,
        required=True,
        help="Rotational diffusion Dr (used in headers and filenames).",
    )
    parser.add_argument(
        "--Dt",
        type=float,
        default=None,
        help="Translational diffusion Dt (optional, for header only).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Number of particles N (optional, for header only).",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Number of integration steps (optional, for header only).",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=180,
        help="Number of histogram bins for the PDF (default: 180).",
    )
    parser.add_argument(
        "--no-add-zero",
        action="store_true",
        help="Do NOT prepend a pdf(0)=0 point to the PDF output.",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory to write output .dat files (relative to repo root).",
    )

    args = parser.parse_args()

    # parent of scripts/ is the repo root
    repo_root = Path(__file__).resolve().parents[1]
    input_path = (repo_root / args.input).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compute_pdf_from_angles] Repo root : {repo_root}")
    print(f"[compute_pdf_from_angles] Input     : {input_path}")
    print(f"[compute_pdf_from_angles] Out dir   : {out_dir}")

    # Load raw angles from the LAMMPS dump
    angles = load_angles_from_dump(input_path)
    print(f"[compute_pdf_from_angles] Loaded {angles.size} angles.")

    # Build method tag and parameter label for filenames
    method = method_tag(args.method)   # "proj" or "geom"
    origin = "lammps"                  # this script is for LAMMPS numerics
    # for real-valued parameters (like Dr, dt) apply format_param, for integers do simply N_str = str(N)
    Dr_str = format_param(args.Dr)
    dt_str = format_param(args.dt)
    label = f"Dr{Dr_str}_dt{dt_str}"

    angles_name = f"angles_{origin}_{method}_{label}.dat"
    pdf_name    = f"pdf_{origin}_{method}_{label}.dat"

    angles_path = out_dir / angles_name
    pdf_path = out_dir / pdf_name

    # Write raw angles (one per line)
    write_angles_file(
        angles_path,
        angles,
        method=args.method,
        N=args.N,
        Dt=args.Dt,
        Dr=args.Dr,
        dt=args.dt,
        nsteps=args.nsteps,
    )
    print(f"[compute_pdf_from_angles] Wrote angles to {angles_path}")

    # Compute PDF and write it
    theta, pdf = compute_pdf(
        angles,
        nbins=args.nbins,
        add_zero=not args.no_add_zero,
    )
    write_pdf_file(
        pdf_path,
        theta,
        pdf,
        method=args.method,
        N=args.N,
        Dt=args.Dt,
        Dr=args.Dr,
        dt=args.dt,
        nsteps=args.nsteps,
        nbins=args.nbins,
    )
    print(f"[compute_pdf_from_angles] Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
