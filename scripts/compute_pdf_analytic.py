#!/usr/bin/env python3
"""
compute_pdf_analytic.py

Compute analytical single-step PDFs for rotational Brownian motion
based on the formulas in the paper:
Felix Höfling & Arthur V. Straube, Phys. Rev. Research 7, 043034 (2025)  
[open access, URL: https://doi.org/10.1103/wzdn-29p4]

- Exact PDF p_exact(theta; Dr, dt) from Eq. (38)
- Geometric Gaussian scheme p_geom(theta; Dr, dt) from Eq. (41)
- Euler + projection scheme p_proj(theta; Dr, dt) from Eq. (44)

The script evaluates all three for given Dr, dt on a theta-grid in [0, pi]
and writes them to whitespace-separated .dat files in the data/ directory.
For Dr=3.0 and dt=0.1 the files look as

    data/pdf_analyt_proj_Dr3.0_dt0.1.dat
    data/pdf_analyt_geom_Dr3.0_dt0.1.dat
    data/pdf_analyt_exact_Dr3.0_dt0.1.dat

Headers include the method, formula reference, and parameters.

Usage from repo root:

    python scripts/compute_pdf_analytic.py --Dr 1.0 --dt 0.1

You can override the truncation parameters lmax, kmax and
the resolution (the number of points npts):
(defaults: lmax=100, kmax=4, npts=1000)

    python scripts/compute_pdf_analytic.py \
        --Dr 1.0 --dt 0.1 --lmax 150 --kmax 6 --npts 2000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Convenience wrapper for ∫ y dx via trapezoid rule."""
    return float(np.trapz(y, x))


def legendre_series_P(x: np.ndarray, lmax: int) -> np.ndarray:
    """
    Return array P with shape (lmax+1, len(x)), where P[ℓ] = P_ℓ(x).

    Implemented using 3-term recurrence:
        P_0 = 1
        P_1 = x
        (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
    """
    x = np.asarray(x)
    L = lmax + 1
    P = np.empty((L, x.size), dtype=float)
    P[0] = 1.0
    if lmax >= 1:
        P[1] = x
        for n in range(1, lmax):
            P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return P


# ---------------------------------------------------------------------------
# Analytical PDFs
# ---------------------------------------------------------------------------

def p_exact(theta: np.ndarray, Dr: float, dt: float, lmax: int) -> np.ndarray:
    r"""
    Exact polar-angle PDF p(θ; Δt) from Eq. (38):

        p(θ; Δt) = ½ sin θ ∑_{ℓ=0}^{∞} (2ℓ + 1) P_ℓ(cos θ) e^{-ℓ(ℓ+1) D_r Δt}

    Here, the sum is truncated at ℓ_max.
    """
    theta = np.asarray(theta)
    x = np.cos(theta)
    P = legendre_series_P(x, lmax)
    ell = np.arange(lmax + 1)
    w = (2 * ell + 1) * np.exp(-ell * (ell + 1) * Dr * dt)
    S = (w[:, None] * P).sum(axis=0)
    return 0.5 * np.sin(theta) * S


def p_geom(theta: np.ndarray, Dr: float, dt: float, kmax: int) -> np.ndarray:
    r"""
    Geometric Gaussian scheme angle PDF, Eq. (41).

    Implemented as a wrapped Gaussian on [0, π], using image sums and
    truncating at |k| ≤ k_max.

    We follow the notebook implementation:

        p_geom(θ; Δt) = ∑_{k=0}^{∞} (θ + 2πk)/(2D_rΔt) e^{-(θ + 2πk)^2/(4D_rΔt)}
                      + ∑_{k=1}^{∞} (2πk - θ)/(2D_rΔt) e^{-(2πk - θ)^2/(4D_rΔt)}.

    """
    theta = np.asarray(theta)
    out = np.zeros_like(theta, dtype=float)
    denom = 2.0 * Dr * dt

    # Images θ + 2πk, k ≥ 0
    for k in range(0, kmax + 1):
        out += (theta + 2.0 * math.pi * k) / denom * np.exp(
            - (theta + 2.0 * math.pi * k) ** 2 / (4.0 * Dr * dt)
        )

    # Reflected images 2πk − θ, k ≥ 1
    for k in range(1, kmax + 1):
        out += (2.0 * math.pi * k - theta) / denom * np.exp(
            - (2.0 * math.pi * k - theta) ** 2 / (4.0 * Dr * dt)
        )

    return out


def p_proj(theta: np.ndarray, Dr: float, dt: float) -> np.ndarray:
    r"""
    Euler + projection scheme angle PDF, Eq. (44).

    Support is only on [0, π/2):

        p_proj(θ; Δt) = [sin θ / (2 D_r Δt cos^3 θ)]
                        * exp(- (cos^{-2} θ - 1)/(4 D_r Δt)),  0 ≤ θ < π/2

    and p_proj(θ) = 0 for θ ≥ π/2.
    """
    theta = np.asarray(theta)
    out = np.zeros_like(theta, dtype=float)
    mask = theta < (0.5 * math.pi)
    ct = np.cos(theta[mask])
    st = np.sin(theta[mask])
    out[mask] = st / (2.0 * Dr * dt * ct ** 3) * np.exp(
        - (ct ** (-2) - 1.0) / (4.0 * Dr * dt)
    )
    return out


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


def write_pdf_file(
    path: Path,
    theta: np.ndarray,
    pdf: np.ndarray,
    *,
    method_label: str,
    eq_label: str,
    Dr: float,
    dt: float,
    npts: int,
    lmax: int | None,
    kmax: int | None,
) -> None:
    """
    Write angle–PDF pairs to a .dat file with a descriptive header.

    method_label: e.g. "projection", "geometric", "exact"
    eq_label:     e.g. "Eq. (44)", "Eq. (41)", "Eq. (38)"
    """
    with path.open("w") as f:
        f.write(f"# method: {method_label}, analytical formula {eq_label}\n")
        f.write(
            "# Dr={Dr} dt={dt} npts={npts} lmax={lmax} kmax={kmax}\n".format(
                Dr=Dr,
                dt=dt,
                npts=npts,
                lmax=lmax if lmax is not None else "NA",
                kmax=kmax if kmax is not None else "NA",
            )
        )
        f.write("# columns: angle(rad)  pdf(angle)\n")
        for th, p in zip(theta, pdf):
            f.write(f"{th:.9f} {p:.9e}\n")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute analytical PDFs (exact, geometric, projection) "
                    "for rotational Brownian motion."
    )
    parser.add_argument(
        "--Dr",
        type=float,
        required=True,
        help="Rotational diffusion constant Dr.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Time step Δt.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=100,
        help="Legendre truncation lmax for the exact series (Eq. 38).",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=4,
        help="Number of wrapped-image terms for the geometric scheme (Eq. 41).",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=1000,
        help="Number of theta points on [0, π].",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory to write output .dat files (relative to repo root).",
    )

    args = parser.parse_args()

    # Repo root = parent of scripts/
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    Dr = args.Dr
    dt = args.dt
    lmax = args.lmax
    kmax = args.kmax
    npts = args.npts
    tau = Dr * dt

    print(f"[compute_pdf_analytic] Repo root : {repo_root}")
    print(f"[compute_pdf_analytic] Out dir   : {out_dir}")
    print(f"[compute_pdf_analytic] Dr={Dr}, dt={dt}, Dr*dt={tau}")

    # Theta grid
    theta = np.linspace(0.0, math.pi, npts)

    # Evaluate PDFs
    p_exact_vals = p_exact(theta, Dr, dt, lmax)
    p_geom_vals  = p_geom(theta, Dr, dt, kmax)
    p_proj_vals  = p_proj(theta, Dr, dt)

    # Check normalization integrals (for diagnostics)
    norm_exact = trapz(p_exact_vals, theta)
    norm_geom  = trapz(p_geom_vals,  theta)
    norm_proj  = trapz(p_proj_vals,  theta)
    print(
        "[compute_pdf_analytic] Normalization integrals: "
        f"exact={norm_exact:.6f}, geom={norm_geom:.6f}, proj={norm_proj:.6f}"
    )

    # Build filenames
    Dr_str = format_param(Dr)
    dt_str = format_param(dt)
    label = f"Dr{Dr_str}_dt{dt_str}"

    file_exact = out_dir / f"pdf_analyt_exact_{label}.dat"
    file_geom  = out_dir / f"pdf_analyt_geom_{label}.dat"
    file_proj  = out_dir / f"pdf_analyt_proj_{label}.dat"

    # Write files
    write_pdf_file(
        file_exact,
        theta,
        p_exact_vals,
        method_label="exact",
        eq_label="(38)",
        Dr=Dr,
        dt=dt,
        npts=npts,
        lmax=lmax,
        kmax=None,
    )
    print(f"[compute_pdf_analytic] Wrote exact PDF to {file_exact}")

    write_pdf_file(
        file_geom,
        theta,
        p_geom_vals,
        method_label="geometric",
        eq_label="(41)",
        Dr=Dr,
        dt=dt,
        npts=npts,
        lmax=None,
        kmax=kmax,
    )
    print(f"[compute_pdf_analytic] Wrote geometric PDF to {file_geom}")

    write_pdf_file(
        file_proj,
        theta,
        p_proj_vals,
        method_label="projection",
        eq_label="(44)",
        Dr=Dr,
        dt=dt,
        npts=npts,
        lmax=None,
        kmax=None,
    )
    print(f"[compute_pdf_analytic] Wrote projection PDF to {file_proj}")


if __name__ == "__main__":
    main()
