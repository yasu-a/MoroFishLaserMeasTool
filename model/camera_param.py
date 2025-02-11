from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraParam:
    mat: np.ndarray

    def from_2d_to_3d(self, u: float, v: float, a: float, b: float, c: float) \
            -> tuple[float, float, float]:
        # (u, v): source point in 2D space
        # ax + by + cz + 1 = 0: plane which destination point lives
        # solve for x, y, z

        (a_11, a_12, a_13, a_14), (a_21, a_22, a_23, a_24), (a_31, a_32, a_33, a_34) = self.mat
        M = np.array([
            [a_11 - a_31 * u, a_12 - a_32 * u, a_13 - a_33 * u],
            [a_21 - a_31 * v, a_22 - a_32 * v, a_23 - a_33 * v],
            [a, b, c],
        ])
        const = np.array([u - a_14, v - a_24, -1])
        x, y, z = np.linalg.inv(M) @ const
        return x, y, z

    def __post_init__(self):
        assert isinstance(self.mat, np.ndarray), "mat should be a numpy array"
        assert self.mat.ndim == 2, "mat should be 2D"
        assert self.mat.shape == (3, 4), "mat should have shape of (3, 4)"

    def to_json(self):
        return {
            "mat": self.mat.tolist(),
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            mat=np.array(json_data["mat"]),
        )


@dataclass(slots=True)
class CameraParamProfile:
    name: str
    param: CameraParam

    def to_json(self):
        return {
            "name": self.name,
            "param": self.param.to_json(),
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            name=json_data["name"],
            param=CameraParam.from_json(json_data["param"]),
        )
