from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ThreeDBody:
    """Simple 3D body struct used by the animation engine."""

    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
