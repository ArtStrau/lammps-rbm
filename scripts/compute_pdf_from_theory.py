#!/usr/bin/env python3
"""
compute_pdf_from_theory.py

Compute rotation-angle single-step probability density functions (PDFs) from theory
for fully overdamped rotational Brownian motion (RBM) on

- sphere (3D RBM, manifold = sphere; orientation space S^2), or
- circle (2D RBM, manifold = circle; orientation space S^1)

for exact solution, geometric and projection integrators.

For a given Dr, formulas for PDFs p(θ; Δt) for the sphere (3D) are taken from:
Felix Höfling & Arthur V. Straube, Phys. Rev. Research 7, 043034 (2025)
(open access: https://doi.org/10.1103/wzdn-29p4).
Here:
  θ ∈ [0, π] is the (wrapped and folded) rotation angle,
 Δt is the integration step,
 Dr is the rotational diffusion constant.

On the sphere (3D), we implement the three one-step angle PDFs from the paper:
- exact       (Eq. 38)
- geometric   (Eq. 41)
- projection  (Eq. 44)

For the circle (2D), we implement the corresponding closed-form kernels
- exact
- geometric
- projection
as indicated in this script and a notebook.

For a chosen manifold (sphere|circle), the script evaluates all three one-step angle PDFs
on a theta-grid θ ∈ [0, π] for given Dr, dt and writes them to whitespace-separated .dat
files in the data/ directory.

Output files:
  data/pdf_theory_<manifold>_exact_Dr<Dr>_dt<dt>.dat
  data/pdf_theory_<manifold>_geometric_Dr<Dr>_dt<dt>.dat
  data/pdf_theory_<manifold>_projection_Dr<Dr>_dt<dt>.dat

Headers include the manifold, method and parameters.

Usage from repo root:

    python scripts/compute_pdf_from_theory.py --manifold sphere --Dr 1.0 --dt 0.1
    python scripts/compute_pdf_from_theory.py --manifold circle --Dr 3.0 --dt 0.1

You can override truncation parameters lmax, kmax and
the resolution (the number of points npts):
(defaults: lmax=100, kmax=4, npts=1000):

    python scripts/compute_pdf_from_theory.py \
        --manifold sphere --Dr 1.0 --dt 0.1 --lmax 150 --kmax 6 --npts 2000

Notes:
- For sphere (3D), lmax truncates the Legendre series (exact kernel),
- For circle (2D), lmax truncates the cosine series (exact kernel),
- kmax controls the number of wrapped-image terms in the (k-sum) kernels:
  sphere geometric (Eq. 41) and circle geometric/exact PDFs.
"""

import argparse
import math
from pathlib import Path

import numpy as np

LOGTAG    = "compute_pdf_from_theory"  # prefix in console logs
OUTPREFIX = "pdf_theory"               # output filename prefix

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def legendre_series_P(x: np.ndarray, lmax: int) -> np.ndarray:
    """
    Compute Legendre polynomials P_ℓ(x) for l = 0..lmax, for a vector x.
    Returns an array P with shape (lmax+1, x.size).

    Implemented using 3-term recurrence:
        P_0 = 1
        P_1 = x
        (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
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
# Theory PDFs: sphere (3D)
# ---------------------------------------------------------------------------

def p_sphere_exact(theta: np.ndarray, Dr: float, dt: float, lmax: int) -> np.ndarray:
    r"""
    Exact rotation-angle single-step PDF p(θ; Δt) on S^2 (PRR Eq. 38).

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


def p_sphere_geom(theta: np.ndarray, Dr: float, dt: float, kmax: int) -> np.ndarray:
    r"""
    Geometric Gaussian scheme (rotation-angle single-step) PDF on S^2 (PRR Eq. 41),
    implemented as a wrapped Gaussian on [0, π] using an image sum truncated at |k| ≤ k_max.

        p_geom(θ; Δt) = Σ_{k=0}^{∞} (θ + 2πk)/(2D_rΔt) e^{-(θ + 2πk)^2/(4D_rΔt)}
                      + Σ_{k=1}^{∞} (2πk - θ)/(2D_rΔt) e^{-(2πk - θ)^2/(4D_rΔt)}

    In code we evaluate both sums symmetrically for k = -kmax..kmax with the
    appropriate terms.
    """
    theta = np.asarray(theta)
    out = np.zeros_like(theta, dtype=float)
    pref = 1.0 / (2.0 * Dr * dt)

    # First sum: images θ + 2πk, k ≥ 0
    for k in range(0, kmax + 1):
        z = theta + 2.0 * math.pi * k
        out += pref * z * np.exp(-z * z / (4.0 * Dr * dt))

    # Second sum: eflected images -θ + 2πk, k ≥ 1
    for k in range(1, kmax + 1):
        z = -theta + 2.0 * math.pi * k
        out += pref * z * np.exp(-z * z / (4.0 * Dr * dt))

    return out


def p_sphere_proj(theta: np.ndarray, Dr: float, dt: float) -> np.ndarray:
    r"""
    Euler + projection scheme (rotation-angle single-step) PDF on S^2 (PRR Eq. 44):

        p_proj(θ; Δt) = sin θ / (2 D_r Δt cos^3 θ)
                        * exp(-(sec^2 θ - 1)/(4 D_r Δt)),  0 ≤ θ < π/2
    and p_proj(θ; Δt) = 0,  θ ≥ π/2
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
# Theory PDFs: circle (2D)
# ---------------------------------------------------------------------------

def p_circle_exact(theta: np.ndarray, Dr: float, dt: float, lmax: int) -> np.ndarray:
    r"""
    Exact rotation-angle single-step PDF p(θ; Δt) on S^1, θ ∈ [0, π], in cosine-series form:

        p(θ; Δt) = 1/π * [ 1 + 2 Σ_{ℓ=1}^{∞} exp(-ℓ^2 D_r Δt) cos(ℓ θ) ].

    Here, the sum is truncated at ℓ_max.
    """
    theta = np.asarray(theta)
    ell = np.arange(1, lmax + 1)
    # broadcast: (ell, theta)
    terms = np.exp(-(ell * ell)[:, None] * Dr * dt) * np.cos(ell[:, None] * theta[None, :])
    return (1.0 / math.pi) * (1.0 + 2.0 * terms.sum(axis=0))


def p_circle_geom(theta: np.ndarray, Dr: float, dt: float, kmax: int) -> np.ndarray:
    r"""
    Geometric rotation-angle one-step kernel (PDF) on S^1 (torque-free benchmark).
    In 2D, the geometric kernel coincides with the exact kernel, and it is convenient
    to evaluate it via an explicit wrapped-Gaussian (image) sum:

        p(θ; Δt) = (2 / sqrt(4π D_r Δt)) Σ_{k∈Z} exp(-(θ + 2πk)^2 / (4 D_r Δt)).

    In code, we truncate the image sum to k = -kmax..kmax.
    """
    theta = np.asarray(theta)
    pref = 2.0 / math.sqrt(4.0 * math.pi * Dr * dt)

    ks = np.arange(-kmax, kmax + 1)
    z = theta[None, :] + 2.0 * math.pi * ks[:, None]
    return pref * np.exp(-(z * z) / (4.0 * Dr * dt)).sum(axis=0)


def p_circle_proj(theta: np.ndarray, Dr: float, dt: float) -> np.ndarray:
    r"""
    Euler + projection kernel (rotation-angle single-step PDF) on S^1,
    folded to θ = |Δφ| ∈ [0, π].

    For 0 ≤ θ < π/2:
        p_proj(θ; Δt) = (2 / sqrt(4π D_r Δt)) * exp(-tan^2 θ / (4 D_r Δt)) * sec^2 θ
    and p_proj(θ; Δt) = 0 for θ ≥ π/2.
    """
    theta = np.asarray(theta)
    out = np.zeros_like(theta, dtype=float)

    mask = theta < (0.5 * math.pi)
    th = theta[mask]
    out[mask] = (2.0 / math.sqrt(4.0 * math.pi * Dr * dt)) * np.exp(
        -(np.tan(th) ** 2) / (4.0 * Dr * dt)
    ) * (1.0 / (np.cos(th) ** 2))

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
    filename: Path,
    theta: np.ndarray,
    pdf: np.ndarray,
    *,
    method_label: str,
    manifold: str,
    Dr: float,
    dt: float,
    npts: int,
    lmax: int | None,
    kmax: int | None,
) -> None:
    """
    Write a two-column file with a small header.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Theory PDF ({manifold}, {method_label})\n")
        f.write(
            "# params: Dr={Dr}  dt={dt}  npts={npts}  lmax={lmax}  kmax={kmax}\n".format(
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
        description="Compute theory PDFs (exact, geometric, projection) for RBM on sphere or circle."
    )
    parser.add_argument(
        "--manifold",
        choices=["sphere", "circle"],
        default="sphere",
        help="Orientation manifold: sphere (3D / S^2) or circle (2D / S^1).",
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
        help="Truncation ℓ_max for the exact series.",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=4,
        help="Image-sum truncation k_max for wrapped kernels.",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=1000,
        help="Number of theta points in [0, π].",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Output directory (default: data/).",
    )

    args = parser.parse_args()

    # Repo root = parent of scripts/
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifold = args.manifold
    Dr = args.Dr
    dt = args.dt
    lmax = args.lmax
    kmax = args.kmax
    npts = args.npts

    tau = Dr * dt
    print(f"[{LOGTAG}] Repo root : {repo_root}")
    print(f"[{LOGTAG}] Out dir   : {out_dir}")
    print(f"[{LOGTAG}] manifold={manifold};  Dr={Dr}, dt={dt}, Dr*dt={tau}")

    # Theta grid
    theta = np.linspace(0.0, math.pi, npts)

    # Select and evaluate manifold-specific PDFs
    if manifold == "sphere":
        p_exact_vals = p_sphere_exact(theta, Dr=Dr, dt=dt, lmax=lmax)
        p_geom_vals = p_sphere_geom(theta, Dr=Dr, dt=dt, kmax=kmax)
        p_proj_vals = p_sphere_proj(theta, Dr=Dr, dt=dt)
    else:
        p_exact_vals = p_circle_exact(theta, Dr=Dr, dt=dt, lmax=lmax)
        # In 2D exact = geometric; we provide a separate k-sum implementation for geometric
        p_geom_vals = p_circle_geom(theta, Dr=Dr, dt=dt, kmax=kmax)
        p_proj_vals = p_circle_proj(theta, Dr=Dr, dt=dt)

    # Normalization check (numerical quadrature on the theta-grid)
    n_exact = float(np.trapz(p_exact_vals, theta))
    n_geom = float(np.trapz(p_geom_vals, theta))
    n_proj = float(np.trapz(p_proj_vals, theta))
    print(
        f"[{LOGTAG}] Normalization integrals: "
        f"exact={n_exact:.10f}  geom={n_geom:.10f}  proj={n_proj:.10f}"
    )

    # Build filenames
    Dr_str = format_param(Dr)
    dt_str = format_param(dt)
    file_exact = out_dir / f"{OUTPREFIX}_{manifold}_exact_Dr{Dr_str}_dt{dt_str}.dat"
    file_geom = out_dir / f"{OUTPREFIX}_{manifold}_geometric_Dr{Dr_str}_dt{dt_str}.dat"
    file_proj = out_dir / f"{OUTPREFIX}_{manifold}_projection_Dr{Dr_str}_dt{dt_str}.dat"

    # Data output
    write_pdf_file(
        file_exact, theta, p_exact_vals,
        method_label="exact", manifold=manifold, Dr=Dr, dt=dt, npts=npts, lmax=lmax, kmax=None
    )
    print(f"[{LOGTAG}] Wrote exact PDF to {file_exact}")

    write_pdf_file(
        file_geom, theta, p_geom_vals,
        method_label="geometric", manifold=manifold, Dr=Dr, dt=dt, npts=npts, lmax=None, kmax=kmax
    )
    print(f"[{LOGTAG}] Wrote geometric PDF to {file_geom}")

    write_pdf_file(
        file_proj, theta, p_proj_vals,
        method_label="projection", manifold=manifold, Dr=Dr, dt=dt, npts=npts, lmax=None, kmax=None
    )
    print(f"[{LOGTAG}] Wrote projection PDF to {file_proj}")


if __name__ == "__main__":
    main()
