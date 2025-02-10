from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraParamProfile:
    name: str
    mat: np.ndarray

    def __post_init__(self):
        assert isinstance(self.mat, np.ndarray), "mat should be a numpy array"
        assert self.mat.ndim == 2, "mat should be 2D"
        assert self.mat.shape == (3, 4), "mat should have shape of (3, 4)"

    def to_json(self):
        return {
            "name": self.name,
            "mat": self.mat.tolist(),
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            name=json_data["name"],
            mat=np.array(json_data["mat"]),
        )
