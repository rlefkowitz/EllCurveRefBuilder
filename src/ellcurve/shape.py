import numpy as np
from typing import List


class Shape:

    ObjectToWorld: List[np.ndarray]
    WorldToObject: List[np.ndarray]
    reverseOrientation: bool
    transformSwapsHandedness: bool

    def __init__(self, ObjectToWorld, WorldToObject, reverseOrientation):
        self.ObjectToWorld = ObjectToWorld
        self.WorldToObject = WorldToObject
        self.reverseOrientation = reverseOrientation
        self.transformSwapsHandedness = np.linalg.det(
            ObjectToWorld[0][:3, :3]) < 0
