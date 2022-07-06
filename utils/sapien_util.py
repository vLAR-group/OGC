"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import numpy as np
from pyquaternion import Quaternion


class Isometry:
    def __init__(self, q: Quaternion = None, t: np.ndarray = None):
        if q is None:
            q = Quaternion()
        if t is None:
            t = np.zeros(3)
        assert t.shape[0] == 3 and t.ndim == 1
        self.q = q
        self.t = t

    def __repr__(self):
        return f"Isometry: t = {self.t}, q = {self.q}"

    @property
    def rotation(self):
        return Isometry(q=self.q)

    @property
    def matrix(self):
        mat = self.q.transformation_matrix
        mat[0:3, 3] = self.t
        return mat

    @staticmethod
    def from_matrix(mat, t_component=None):
        assert isinstance(mat, np.ndarray)
        if t_component is None:
            assert mat.shape == (4, 4)
            return Isometry(q=Quaternion(matrix=mat), t=mat[0:3, 3])
        else:
            assert mat.shape == (3, 3)
            assert t_component.shape == (3,)
            return Isometry(q=Quaternion(matrix=mat), t=t_component)

    @staticmethod
    def random():
        return Isometry(q=Quaternion.random(), t=np.random.random((3,)))

    def inv(self):
        qinv = self.q.inverse
        return Isometry(q=qinv, t=-(qinv.rotate(self.t)))

    def dot(self, right):
        return Isometry(q=(self.q * right.q), t=(self.q.rotate(right.t) + self.t))

    def __matmul__(self, other):
        if isinstance(other, Isometry):
            return self.dot(other)
        if type(other) != np.ndarray or other.ndim == 1:
            return self.q.rotate(other) + self.t
        else:
            return other @ self.q.rotation_matrix.T + self.t[np.newaxis, :]
