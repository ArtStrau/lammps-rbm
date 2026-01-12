"""RBM notebook utilities.

This is an internal, lightweight module bundle currently used by the RBM benchmark
notebooks. It is not meant to be installed as a standalone package; the notebooks
add the repo root to `sys.path` so these modules can be imported.
"""

from .io import fmt_float, load_xy
from .plot import plot_compare_pdf

__all__ = ["fmt_float", "load_xy", "plot_compare_pdf"]
