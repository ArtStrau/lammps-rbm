"""I/O + formatting helpers for RBM benchmark notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fmt_float(x: float) -> str:
    """Match the convention on numbers in the filename."""
    # --- current behavior: keep one decimal for integer-like values ---
    if float(x).is_integer():
        return f"{x:.1f}"      # 3, 3.0 or 3.00 -> "3.0"
    # compact representation for non-integer values
    return f"{x:.8g}"          # 0.01 -> "0.01"
    # ------------------------------------------------------------------
    # Alternative behavior (shorter; uncomment and remove the code above):
    # works as 3, 3.0 or 3.00 -> "3" otherwise 0.01 -> "0.01"
    # return f"{x:.8g}"     # compact representation applies to all values


def load_xy(path: Path):
    """Load 2-column .dat with '#' comments: returns (theta, pdf)."""
    theta, pdf = np.loadtxt(path, comments="#", unpack=True)
    return theta, pdf
