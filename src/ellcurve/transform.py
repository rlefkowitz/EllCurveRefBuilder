from typing import List, Optional
import numpy as np
from math import cos, radians, sin, sqrt
from util import gamma, normalize, Ray, T, zero3f


class Transform:

    m: np.ndarray
    minv: np.ndarray

    def __init__(self, m: np.ndarray, minv: np.ndarray):
        self.m = m
        self.minv = minv

    def __applyTo3f__(self, p: np.ndarray, pError: Optional[List[np.ndarray]] = None) -> np.ndarray:
        m: np.ndarray = self.m
        x: float = p[0]
        y: float = p[1]
        z: float = p[2]
        xp: float = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
        yp: float = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
        zp: float = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
        wp: float = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]

        if pError:
            xAbsSum: float = (abs(m[0][0] * x) + abs(m[0][1] * y) +
                              abs(m[0][2] * z) + abs(m[0][3]))
            yAbsSum: float = (abs(m[1][0] * x) + abs(m[1][1] * y) +
                              abs(m[1][2] * z) + abs(m[1][3]))
            zAbsSum: float = (abs(m[2][0] * x) + abs(m[2][1] * y) +
                              abs(m[2][2] * z) + abs(m[2][3]))

            pError[0] = gamma(3) * np.array([xAbsSum, yAbsSum, zAbsSum])

        if wp == 1:
            return np.array([xp, yp, zp])
        else:
            return np.array([xp, yp, zp]) / wp

    def __applyToRay__(self, r: Ray) -> Ray:
        oError: List[np.ndarray] = [zero3f()]
        o: np.ndarray = self(r.o, oError)
        d: np.ndarray = self(r.d)

        lengthSquared: float = np.inner(d, d)
        tMax: float = r.tMax
        if lengthSquared > 0:
            dt: float = np.dot(np.abs(d), oError[0]) / lengthSquared
            o += d * dt
            tMax -= dt
        return Ray(o, d, tMax, r.time, r.medium)

    def __call__(self, x: T, pError: Optional[List[np.ndarray]] = None) -> T:
        if isinstance(x, Ray):
            return self.__applyToRay__(x)
        return self.__applyTo3f__(x, pError)


def Inverse(x: Transform) -> Transform:
    return Transform(x.minv, x.m)


def CoordinateSystem(v1: np.ndarray, v2: List[np.ndarray], v3: List[np.ndarray]):
    if abs(v1[0]) > abs(v1[1]):
        v2[0] = np.array([-v1[2], 0, v1[0]]) / \
            np.sqrt(v1[0] * v1[0] + v1[2] * v1[2])
    else:
        v2[0] = np.array([0, v1[2], -v1[1]]) / \
            sqrt(v1[0] * v1[0] + v1[2] * v1[2])
    v3[0] = np.cross(v1, v2[0])


def LookAt(pos: np.ndarray, look: np.ndarray, up: np.ndarray) -> Transform:
    cameraToWorld: np.ndarray = np.identity(4)
    cameraToWorld[0][3] = pos[0]
    cameraToWorld[1][3] = pos[1]
    cameraToWorld[2][3] = pos[2]
    cameraToWorld[3][3] = 1.
    dir: np.ndarray = normalize(look - pos)
    right: np.ndarray = normalize(np.cross(normalize(up), dir))
    newUp: np.ndarray = np.cross(dir, right)
    cameraToWorld[0][0] = right[0]
    cameraToWorld[1][0] = right[1]
    cameraToWorld[2][0] = right[2]
    cameraToWorld[3][0] = 0.
    cameraToWorld[0][1] = newUp[0]
    cameraToWorld[1][1] = newUp[1]
    cameraToWorld[2][1] = newUp[2]
    cameraToWorld[3][1] = 0.
    cameraToWorld[0][2] = dir[0]
    cameraToWorld[1][2] = dir[1]
    cameraToWorld[2][2] = dir[2]
    cameraToWorld[3][2] = 0.

    return Transform(np.linalg.inv(cameraToWorld), cameraToWorld)


def RotateZ(theta: float) -> Transform:
    sinTheta: float = sin(radians(theta))
    cosTheta: float = cos(radians(theta))
    m: np.ndarray = np.array([[cosTheta, -sinTheta, 0., 0.],
                              [sinTheta, cosTheta, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return Transform(m, m.transpose())


def Rotate(theta: float, axis: np.ndarray) -> Transform:
    a: np.ndarray = normalize(axis)
    sinTheta: float = sin(radians(theta))
    cosTheta: float = cos(radians(theta))
    m: np.ndarray = np.identity(4)

    # Compute rotation of first basis vector
    m[0][0] = a[0] * a[0] + (1. - a[0] * a[0]) * cosTheta
    m[0][1] = a[0] * a[1] + (1. - cosTheta) - a[2] * sinTheta
    m[0][2] = a[0] * a[2] + (1. - cosTheta) + a[1] * sinTheta

    # Compute rotations of second and third basis vectors
    m[1][0] = a[0] * a[1] + (1. - cosTheta) + a[2] * sinTheta
    m[1][1] = a[1] * a[1] + (1. - a[1] * a[1]) * cosTheta
    m[1][2] = a[1] * a[2] + (1. - cosTheta) - a[0] * sinTheta

    m[2][0] = a[0] * a[2] + (1. - cosTheta) - a[1] * sinTheta
    m[2][1] = a[1] * a[2] + (1. - cosTheta) + a[0] * sinTheta
    m[2][2] = a[2] * a[2] + (1. - a[2] * a[2]) * cosTheta

    return Transform(m, m.transpose())
