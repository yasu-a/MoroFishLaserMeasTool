from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LaserParamProfile:
    name: str
    vec: np.ndarray

    def __post_init__(self):
        assert isinstance(self.vec, np.ndarray), "vec should be a numpy array"
        assert self.vec.ndim == 1, "vec should be 1D"
        if self.vec.size == 3:
            self.vec = np.array([*self.vec, 1])
        assert self.vec.size == 4, "vec should have 4 elements"

    def to_json(self):
        return {
            "name": self.name,
            "vec": self.vec.tolist(),
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            name=json_data["name"],
            vec=np.array(json_data["vec"]),
        )
