import os
from curve.curve import Curve
from typing import List

class EllCurveRefBuilder:

    curves: List[Curve]
    uRes: int
    vRes: int

    def __init__(self, curves: List[Curve], uRes: int, vRes: int):
        self.curves = curves
        self.uRes = uRes
        self.vRes = vRes

    def build(self, output_path: str):
        os.remove(output_path)
        f = open(output_path)
