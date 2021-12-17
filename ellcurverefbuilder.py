from src.builder import EllCurveRefBuilder
from src.ellcurve.ellcurve import Curve
from typing import List
import numpy as np


def makeCurves() -> List[Curve]:
    points = [
        [-16., 0., 0.], [-5.333333, 0., 0.], [5.333333, 0., 0.], [16., 0., 0.]
    ]
    curve = CreateCurveShape([np.identity(4)], [np.identity(4)], False, {
            "type": "cylinder",
            "width": 3.,
            # "ecc": 0.99,
            # "ang0": 0.,
            # "ang1": 360.,
            "P": [np.array(p) for p in points]
        })
    return curve

builder = EllCurveRefBuilder()