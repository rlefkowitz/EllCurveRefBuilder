from math import cos, radians, sin, sqrt

import numpy as np
from typing import Any, List, TypeVar, Union
from shape import Shape


T = TypeVar('T')


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def zero3f() -> np.ndarray:
    return np.array([0., 0., 0.])


def gamma(n: int) -> float:
    MachineEpsilon: float = 5.960464478e-08
    return (n * MachineEpsilon) / (1. - n * MachineEpsilon)


class Ray:

    o: np.ndarray
    d: np.ndarray
    tMax: float
    time: float
    medium: Any

    def __init__(self, o, d, tMax=1e20, time=0., medium=None):
        self.o = o
        self.d = d
        self.tMax = tMax
        self.time = time
        self.medium = medium

    def __call__(self, t: float) -> np.ndarray:
        return self.o + self.d * t


class Interaction:

    p: np.ndarray
    time: float
    pError: np.ndarray
    wo: np.ndarray
    n: np.ndarray
    mediumInterface: Any

    def __init__(self, p=None, n=None, pError=None, wo=None, time=None, mediumInterface=None):
        if p is None:
            self.time = 0
        else:
            self.p = p
            self.time = time
            self.pError = pError
            self.wo = wo
            self.n = n
            self.mediumInterface = mediumInterface


class Shading:

    n: np.ndarray
    dpdu: np.ndarray
    dpdv: np.ndarray
    dndu: np.ndarray
    dndv: np.ndarray

    def __init__(self):
        self.n = zero3f()
        self.dpdu = zero3f()
        self.dpdv = zero3f()
        self.dndu = zero3f()
        self.dndv = zero3f()


class SurfaceInteraction(Interaction):

    uv: np.ndarray
    dpdu: np.ndarray
    dpdv: np.ndarray
    dndu: np.ndarray
    dndv: np.ndarray
    shape: List[Shape]
    shading: Shading

    def __init__(self, p=None, pError=None, uv=None, wo=None, dpdu=None, dpdv=None, dndu=None, dndv=None, time=None, shape=None):
        n: np.ndarray = normalize(np.cross(dpdu, dpdv))
        super().__init__(p, n, pError, wo, time, None)
        self.uv = uv
        self.dpdu = dpdu
        self.dpdv = dpdv
        self.dndu = dndu
        self.dndv = dndv
        self.shape = shape

        self.shading = Shading()
        self.shading.n = self.n
        self.shading.dpdu = self.dpdu
        self.shading.dpdv = self.dpdv
        self.shading.dndu = self.dndu
        self.shading.dndv = self.dndv

        if shape and shape[0] and (shape[0].reverseOrientation ^ shape[0].transformSwapsHandedness):
            self.n *= -1.
            self.shading.n *= -1.


def Lerp(t: float, v1: T, v2: T) -> T:
    return (1 - t) * v1 + t * v2
