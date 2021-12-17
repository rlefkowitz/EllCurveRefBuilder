from typing import Any, List

from shape import Shape
from math import acos, cos, radians, sin, sqrt, tan
import numpy as np

from transform import CoordinateSystem, Inverse, LookAt, Rotate, RotateZ, Transform

from util import Ray, SurfaceInteraction, Lerp, normalize, zero3f
from bounds import Bounds3f, Union, Expand, Overlaps


def BlossomBezier(p, u0, u1, u2):
    a = [Lerp(u0, p[0], p[1]), Lerp(u0, p[1], p[2]), Lerp(u0, p[2], p[3])]
    b = [Lerp(u1, a[0], a[1]), Lerp(u1, a[1], a[2])]
    return Lerp(u2, b[0], b[1])


def SubdivideBezier(cp, cpSplit):
    cpSplit[0] = cp[0]
    cpSplit[1] = (cp[0] + cp[1]) / 2.
    cpSplit[2] = (cp[0] + 2. * cp[1] + cp[2]) / 4.
    cpSplit[3] = (cp[0] + 3. * cp[1] + 3. * cp[2] + cp[3]) / 8.
    cpSplit[4] = (cp[1] + 2. * cp[2] + cp[3]) / 4.
    cpSplit[5] = (cp[2] + cp[3]) / 2.
    cpSplit[6] = cp[3]


def EvalBezier(cp, u, deriv=None):
    cp1 = [Lerp(u, cp[0], cp[1]), Lerp(u, cp[1], cp[2]), Lerp(u, cp[2], cp[3])]
    cp2 = [Lerp(u, cp1[0], cp1[1]), Lerp(u, cp1[1], cp1[2])]
    if deriv:
        deriv[0] = 3. * (cp2[1] - cp2[0])
    return Lerp(u, cp2[0], cp2[1])

def CreateCurve(o2w, w2o, reverseOrientation, c, w0, w1, e0, e1, a0, a1, norm, splitDepth):
    segments = []
    common = CurveCommon(c, w0, w1, e0, e1, a0, a1, norm)
    nSegments = 1 << splitDepth
    segments = [None] * nSegments
    for i in range(nSegments):
        uMin = i / float(nSegments)
        uMax = (i + 1) / float(nSegments)
        segments[i] = Curve(o2w, w2o, reverseOrientation, common, uMin, uMax)
    return segments


def CreateCurveShape(o2w, w2o, reverseOrieintation, params):


    width = params["width"] if "width" in params else 1.
    width0 = params["width0"] if "width0" in params else width
    width1 = params["width1"] if "width1" in params else width

    ecc = params["ecc"] if "ecc" in params else 0.
    ecc0 = params["ecc0"] if "ecc0" in params else ecc
    ecc1 = params["ecc1"] if "ecc1" in params else ecc

    ang = params["ang"] if "ang" in params else 0.
    ang0 = params["ang0"] if "ang0" in params else ang
    ang1 = params["ang1"] if "ang1" in params else ang
    cp = params["P"]

    curve = CreateCurve(o2w, w2o, reverseOrieintation, cp, width0, width1, ecc0, ecc1, ang0, ang1)
    # width = params["width"] if "width" in params else 1.
    # width0 = params["width0"] if "width0" in params else width
    # width1 = params["width1"] if "width1" in params else width

    # ecc = params["ecc"] if "ecc" in params else 0.
    # ecc0 = params["ecc0"] if "ecc0" in params else ecc
    # ecc1 = params["ecc1"] if "ecc1" in params else ecc

    # ang = params["ang"] if "ang" in params else 0.
    # ang0 = params["ang0"] if "ang0" in params else ang
    # ang1 = params["ang1"] if "ang1" in params else ang

    # degree = params["degree"] if "degree" in params else 3

    # basis = params["basis"] if "basis" in params else "bezier"

    # ncp = [0]
    # cp = params["P"]
    # ncp[0] = len(cp)
    # nSegments = 0
    # if basis == "bezier":
    #     nSegments = int((ncp[0] - 1) / degree)
    # else:
    #     nSegments = int(ncp[0] - degree)

    # nnorm = [0]
    # n = params["N"] if "N" in params else None
    # nnorm[0] = len(n) if n else 0
    # sd = params["splitdepth"] if "splitdepth" in params else int(
    #     params["splitdepth"] if "splitdepth" in params else 3)

    # curves = []
    # cpBase = cp
    # for seg in range(nSegments):
    #     segCpBezier = [None] * 4

    #     if basis == "bezier":
    #         if degree == 2:
    #             segCpBezier[0] = cpBase[0]
    #             segCpBezier[1] = Lerp(2. / 3., cpBase[0], cpBase[1])
    #             segCpBezier[2] = Lerp(1. / 3., cpBase[1], cpBase[2])
    #             segCpBezier[3] = cpBase[2]
    #         else:
    #             for i in range(4):
    #                 segCpBezier[i] = cpBase[i]
    #         cpBase = cpBase[degree:]
    #     else:
    #         if degree == 2:
    #             p01 = cpBase[0]
    #             p12 = cpBase[1]
    #             p23 = cpBase[2]

    #             p11 = Lerp(0.5, p01, p12)
    #             p22 = Lerp(0.5, p12, p23)

    #             segCpBezier[0] = p11
    #             segCpBezier[1] = Lerp(2. / 3., p11, p12)
    #             segCpBezier[2] = Lerp(1. / 3., p12, p22)
    #             segCpBezier[3] = p22
    #         else:
    #             p012 = cpBase[0]
    #             p123 = cpBase[1]
    #             p234 = cpBase[2]
    #             p345 = cpBase[3]

    #             p122 = Lerp(2. / 3., p012, p123)
    #             p223 = Lerp(1. / 3., p123, p234)
    #             p233 = Lerp(2. / 3., p123, p234)
    #             p334 = Lerp(1. / 3., p234, p345)

    #             p222 = Lerp(0.5, p122, p223)
    #             p333 = Lerp(0.5, p233, p334)

    #             segCpBezier[0] = p222
    #             segCpBezier[1] = p223
    #             segCpBezier[2] = p233
    #             segCpBezier[3] = p333
    #         cpBase += 1

    #     c = CreateCurve(o2w, w2o, reverseOrieintation, segCpBezier,
    #                     Lerp(float(seg) / float(nSegments), width0, width1),
    #                     Lerp(float(seg + 1) / float(nSegments), width0, width1),
    #                     Lerp(float(seg) / float(nSegments), ecc0, ecc1),
    #                     Lerp(float(seg + 1) / float(nSegments), ecc0, ecc1),
    #                     Lerp(float(seg) / float(nSegments), ang0, ang1),
    #                     Lerp(float(seg + 1) / float(nSegments), ang0, ang1),
    #                     n[seg:] if n else None, sd)
    #     curves += c
    # return curves


class CurveCommon:

    cpObj: List[np.ndarray]
    width: List[float]
    ecc: List[float]
    ang: List[float]

    def __init__(self, c, width0, width1, ecc0, ecc1, ang0, ang1):
        self.cpObj = [c[0], c[1], c[2], c[3]]
        self.width = [width0, width1]
        self.ecc = [ecc0, ecc1]
        self.ang = [ang0, ang1]


class Curve(Shape):

    common: CurveCommon
    uMin: float
    uMax: float

    def __init__(self, ObjectToWorld, WorldToObject, reverseOrientation, common, uMin, uMax):
        super().__init__(ObjectToWorld, WorldToObject, reverseOrientation)
        self.common = common
        self.uMin = uMin
        self.uMax = uMax

    def ObjectBound(self):
        cpObj = [0] * 4
        cpObj[0] = BlossomBezier(
            self.common.cpObj, self.uMin, self.uMin, self.uMin)
        cpObj[1] = BlossomBezier(
            self.common.cpObj, self.uMin, self.uMin, self.uMax)
        cpObj[2] = BlossomBezier(
            self.common.cpObj, self.uMin, self.uMax, self.uMax)
        cpObj[3] = BlossomBezier(
            self.common.cpObj, self.uMax, self.uMax, self.uMax)
        b = Union(Bounds3f(cpObj[0], cpObj[1]), Bounds3f(cpObj[2], cpObj[3]))
        width = [
            Lerp(self.uMin, self.common.width[0], self.common.width[1]),
            Lerp(self.uMax, self.common.width[0], self.common.width[1])
        ]
        return Expand(b, np.max(width[0], width[1]) * 0.5)

    def getVertices(self, uRes: int, vRes: int) -> List[np.ndarray]:
        res = []
        for i in range(uRes):
            u: float = 1.0 / float(uRes - 1)
            tangent: List[np.ndarray] = [zero3f()]
            center: np.ndarray = EvalBezier(self.common.cpObj, u, tangent)
            for j in range(vRes):
                pass
        pass