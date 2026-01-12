"""Plotting helpers for RBM angle-PDF comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib.axes import Axes

from .io import fmt_float, load_xy


# --- default color palette (matches existing notebooks) ---
COL_EXACT: str = "#ed1717"   # analytic, exact (red)
COL_PROJ: str = "#ffb000"    # analytic, projection (yellow)
COL_GEOM: str = "#0295FF"    # analytic, geometric (light blue)

# Marker face colors (LAMMPS results)
COL_PROJ_MARK: str = COL_PROJ
COL_GEOM_MARK: str = COL_GEOM


def _require_file(path: Path) -> None:
    """Raise a clear error if a required data file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")


def plot_compare_pdf(
    ax: Axes,
    *,
    manifold: str,
    data_dir: Path,
    Dr: float,
    dt: float,
    # default values; override when calling if required:
    xlim: Tuple[float, float] = (0.0, 1.0),               # axis settings for theta/pi
    lw_exact: float = 7.0,
    lw_proj: float = 3.0,
    lw_geom: float = 2.0,
    step_proj: int = 2,                                   # marker thinning
    step_geom: int = 2,
    ms_proj: float = 4.8,
    ms_geom: float = 4.5,
    col_exact: str = COL_EXACT,
    col_proj: str = COL_PROJ,
    col_geom: str = COL_GEOM,
    col_proj_mark: str = COL_PROJ_MARK,
    col_geom_mark: str = COL_GEOM_MARK,
) -> None:
    """Plot analytic vs LAMMPS PDFs for one (Dr, dt) on the given axes.

    The function expects files produced by your pipeline with names of the form:

        pdf_theory_{manifold}_{exact|projection|geometric}_Dr{Dr}_dt{dt}.dat
        pdf_lammps_{manifold}_{projection|geometric}_Dr{Dr}_dt{dt}.dat

    where `Dr` and `dt` are formatted via :func:`rbm.io.fmt_float`.

    Parameters
    ----------
    ax
        Matplotlib axes to draw into.
    manifold
        "sphere" (3D) or "circle" (2D). The value is used only in filenames.
    data_dir
        Directory containing the produced .dat files.
    Dr, dt
        Rotational diffusion constant and time step.
    """
    Dr_str = fmt_float(Dr)
    dt_str = fmt_float(dt)

    # --- file paths (must match the produced filenames) ---
    fa_exact = data_dir / f"pdf_theory_{manifold}_exact_Dr{Dr_str}_dt{dt_str}.dat"
    fa_proj  = data_dir / f"pdf_theory_{manifold}_projection_Dr{Dr_str}_dt{dt_str}.dat"
    fa_geom  = data_dir / f"pdf_theory_{manifold}_geometric_Dr{Dr_str}_dt{dt_str}.dat"

    fl_proj  = data_dir / f"pdf_lammps_{manifold}_projection_Dr{Dr_str}_dt{dt_str}.dat"
    fl_geom  = data_dir / f"pdf_lammps_{manifold}_geometric_Dr{Dr_str}_dt{dt_str}.dat"

    for p in (fa_exact, fa_proj, fa_geom, fl_proj, fl_geom):
        _require_file(p)

    # --- analytic curves (lines) ---
    th, p = load_xy(fa_exact)
    ax.plot(th / np.pi, p, color=col_exact, lw=lw_exact, label="Analytic, exact")

    th, p = load_xy(fa_proj)
    ax.plot(th / np.pi, p, color=col_proj, lw=lw_proj, label="Analytic, projection")

    th, p = load_xy(fa_geom)
    ax.plot(th / np.pi, p, color=col_geom, lw=lw_geom, label="Analytic, geometric")

    # --- LAMMPS projection (markers) ---
    x, p = load_xy(fl_proj)
    ax.plot(
        x[::step_proj] / np.pi,
        p[::step_proj],
        linestyle="none",
        marker="o",
        markersize=ms_proj,
        markeredgewidth=0.5,
        color="black",
        markerfacecolor=col_proj_mark,
        label="LAMMPS, projection",
    )

    # --- LAMMPS geometric (markers) ---
    x, p = load_xy(fl_geom)
    ax.plot(
        x[::step_geom] / np.pi,
        p[::step_geom],
        linestyle="none",
        marker="s",
        markersize=ms_geom,
        markeredgewidth=0.5,
        color="black",
        markerfacecolor=col_geom_mark,
        label="LAMMPS, geometric",
    )

    # --- axes & legend defaults (customize after calling if needed) ---
    ax.set_xlabel(r"$\vartheta/\pi$")
    ax.set_ylabel(r"$p(\vartheta; \Delta t)$")
    ax.set_xlim(*xlim)
    ax.legend(frameon=False)
