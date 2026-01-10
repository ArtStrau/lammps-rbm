#!/usr/bin/env python3
"""
compute_pdf_from_angles.py

Postprocess a LAMMPS dump of per-atom angles and produce

  1) a cleaned angle sample file (one angle per line), and
  2) a normalized probability density function (PDF) over [0, pi].

The script is manifold- (dimension) and method independent: it only
sees angles obtained from LAMMPS simulation of for fully overdamped
rotational Brownian motion (RBM). Specify the manifold (dimension)
for RBM on 

- sphere (3D RBM, manifold = sphere; orientation space S^2), or
- circle (2D RBM, manifold = circle; orientation space S^1)

and the method (intergator scheme) used

- geometric integrator as proposed in the reference or
- projection (Euler + projection) scheme.

so that the output filenames are tagged accordingly.

Reference: Felix Höfling & Arthur V. Straube, Phys. Rev. Research 7, 043034 (2025)
(open access: https://doi.org/10.1103/wzdn-29p4).

Typical usage from the repo root:

    # sphere (3D) + geometric integrator
    python scripts/compute_pdf_from_angles.py \
        --manifold sphere --method geometric \
        --N 10000 --Dt 1.0 --Dr 3.0 --dt 0.1 --nsteps 1000 [--nbins 180]

    # circle (2D) + projection integrator
    python scripts/compute_pdf_from_angles.py \
        --manifold circle --method projection \
        --N 10000 --Dt 1.0 --Dr 3.0 --dt 0.1 --nsteps 1000 [--nbins 180]

Here:
   mandatory: (used to form output file names):
  manifold = sphere | circle,
    method = geometric | projection,
        Dr is the rotational diffusion constant,
        dt is the integration step
   optional:
         N is the number of particles,
        Dt is the translational diffusion constant,
    nsteps is the number of integration steps,
     nbins is the resolution (number of bins, 180 by default).

By default, if you do not pass --input, the script reads from:

    outputs/angles_<manifold>_<method>.raw

Output files:

    data/angles_lammps_<manifold>_<method>_Dr<Dr>_dt<dt>.dat
    data/pdf_lammps_<manifold>_<method>_Dr<Dr>_dt<dt>.dat

use '#' comment headers, which include the manifold, method and parameters,
followed by whitespace-separated columns.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np

LOGTAG = "compute_pdf_from_angles"  # prefix in console logs

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Formatting / output helpers
# ---------------------------------------------------------------------------

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


def compute_pdf(
    angles: np.ndarray,
    nbins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized PDF over [0, pi] from a 1D array of angles.

    Returns (theta, pdf) where theta are bin centers in [0, pi],
    and pdf(theta) is normalized so that the integral over [0, pi] is ~1.
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
    manifold: str,
    N: int | None,
    Dt: float | None,
    Dr: float,
    dt: float,
    nsteps: int | None,
    nbins: int | None,
) -> None:
    """Write angle–PDF pairs with a small header."""
    with path.open("w") as f:
        f.write(f"# Lammps numerics PDF ({manifold}, {method})\n")
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


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute angle PDF from LAMMPS dump of per-atom angles."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to LAMMPS dump with angles (relative to repo root). Default: outputs/angles_<manifold>_<method>.raw",
    )
    parser.add_argument(
        "--manifold",
        choices=["sphere", "circle"],
        default="sphere",
        help="Manifold label for filenames and default input: sphere (3D) or circle (2D).",
    )

    parser.add_argument(
        "--method",
        choices=["projection", "geometric"],
        required=True,
        help="Integrator method label for filenames and headers.",
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
        "--out-dir",
        default="data",
        help="Directory to write output .dat files (relative to repo root).",
    )

    args = parser.parse_args()

    # parent of scripts/ is the repo root
    repo_root = Path(__file__).resolve().parents[1]

    if args.input is None:
        args.input = f"outputs/angles_{args.manifold}_{args.method}.raw"

    input_path = (repo_root / args.input).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LOGTAG}] Repo root : {repo_root}")
    print(f"[{LOGTAG}] Input     : {input_path}")
    print(f"[{LOGTAG}] Out dir   : {out_dir}")

    # Load raw angles from the LAMMPS dump
    angles = load_angles_from_dump(input_path)
    print(f"[{LOGTAG}] Loaded {angles.size} angles.")

    # Build method tag and parameter label for filenames
    method = args.method              # "projection" or "geometric"
    origin = "lammps"                 # this script is for LAMMPS numerics
    # for real-valued parameters (like Dr, dt) apply format_param, for integers do simply N_str = str(N)
    Dr_str = format_param(args.Dr)
    dt_str = format_param(args.dt)
    label = f"Dr{Dr_str}_dt{dt_str}"

    angles_name = f"angles_{origin}_{args.manifold}_{method}_{label}.dat"
    pdf_name    = f"pdf_{origin}_{args.manifold}_{method}_{label}.dat"

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
    print(f"[{LOGTAG}] Wrote angles to {angles_path}")

    # Compute PDF and write it
    theta, pdf = compute_pdf(
        angles,
        nbins=args.nbins,
    )
    # Histogram is piecewise-constant: check normalization via Σ p_i Δθ (not trapz on bin centers)
    dtheta = math.pi / args.nbins
    norm = float(np.sum(pdf) * dtheta)
    print(f"[{LOGTAG}] PDF normalization: {norm:.6f}")
    write_pdf_file(
        pdf_path,
        theta,
        pdf,
        method=args.method,
        manifold=args.manifold,
        N=args.N,
        Dt=args.Dt,
        Dr=args.Dr,
        dt=args.dt,
        nsteps=args.nsteps,
        nbins=args.nbins,
    )
    print(f"[{LOGTAG}] Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
