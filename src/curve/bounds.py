import numpy as np
from typing import Optional, Union


class Bounds3f:

    pMin: np.ndarray
    pMax: np.ndarray

    def __init__(self, p1: Optional[np.ndarray] = None, p2: Optional[np.ndarray] = None):
        if p1 is None:
            self.pMin = np.array([1.0e20, 1.0e20, 1.0e20])
            self.pMax = np.array([-1.0e20, -1.0e20, -1.0e20])
        elif p2 is None:
            self.pMin = np.copy(p1)
            self.pMax = np.copy(p1)
        else:
            self.pMin = np.minimum(p1, p2)
            self.pMax = np.maximum(p1, p2)


def Union(b1: Bounds3f, b2: Union[float, Bounds3f]) -> Bounds3f:
    if isinstance(b2, Bounds3f):
        return Bounds3f(np.minimum(b1.pMin, b2.pMin), np.maximum(b1.pMax, b2.pMin))
    else:
        return Bounds3f(np.minimum(b1.pMin, b2), np.maximum(b1.pMax, b2))


def Expand(b: Bounds3f, delta: float) -> Bounds3f:
    return Bounds3f(b.pMin - np.array([delta, delta, delta]), b.pMax + np.array([delta, delta, delta]))


def Overlaps(b1: Bounds3f, b2: Bounds3f) -> bool:
    return np.all((b1.pMax >= b2.pMin) & (b1.pMin <= b2.pMax))
