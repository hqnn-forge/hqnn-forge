"""
hqnn_forge._warnings
====================
Attribute a library warning to the user's own call, however deep it was raised.
"""

from __future__ import annotations

import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def external_stacklevel() -> int:
    """
    The ``stacklevel`` for a ``warnings.warn`` in the calling function that
    attributes the warning to the first frame outside ``hqnn_forge``.

    A fixed stacklevel is right for one call depth only, and the same
    warning is reached directly, through a classifier's ``__init__`` or
    through a diagnostic.  ``warnings.warn(skip_file_prefixes=...)`` does
    this natively but needs Python 3.12.
    """
    frame = sys._getframe(1)  # the function about to call warnings.warn: stacklevel 1
    level = 1
    while frame.f_back is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR
    ):
        frame = frame.f_back
        level += 1
    return level
