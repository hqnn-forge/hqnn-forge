"""
hqnn_forge
==========
Parameter-efficient Hybrid Quantum Neural Networks for imbalanced tabular classification.

Public API surface — import the most commonly used symbols directly from the top-level
package so user code stays concise:

    from hqnn_forge.encoding  import QuantumEncodingLayer
    from hqnn_forge.models    import HybridBinaryClassifier
    from hqnn_forge.utils     import FocalLoss

Exported symbols
----------------
__version__  The installed package version ("0.1.0-dev" when running from source without an install).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("hqnn-forge")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.1.0-dev"

__all__: list[str] = [
    "__version__",
]
