"""
PyThia performs video prediction for simple physical systems:
given initial video frames, what will a video frame in the future look like?
"""

__version__ = "0.1.0"

from . import demo
from . import data_generation
from . import visualization

__all__ = ["data_generation", "visualization"]
