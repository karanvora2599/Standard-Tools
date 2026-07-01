"""Standard quantitative finance tools for backtesting, analysis, and agent-based trading."""

import logging

__version__ = "0.1.0"

# Library-level NullHandler — callers configure handlers; we never emit by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
