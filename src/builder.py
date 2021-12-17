import os
from ellcurve.ellcurve import Curve
from typing import List

class EllCurveRefBuilder:

    uRes: int
    vRes: int
    curves: List[Curve]

    def __init__(self, curves: List[Curve], uRes: int, vRes: int):
        self.curves = curves
        self.uRes = uRes
        self.vRes = vRes

    def build(self, output_path: str):
        f = open("result")