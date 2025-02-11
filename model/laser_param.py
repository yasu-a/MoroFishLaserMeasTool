from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LaserParam:
    vec: np.ndarray
    error: float

    def __post_init__(self):
        assert isinstance(self.vec, np.ndarray), "vec should be a numpy array"
        assert self.vec.ndim == 1, "vec should be 1D"
        if self.vec.size == 3:
            self.vec = np.array([*self.vec, 1])
        assert self.vec.size == 4, "vec should have 4 elements"

        self.error = float(self.error)

    def to_json(self):
        return {
            "vec": self.vec.tolist(),
            "error": self.error,
        }

    @classmethod
    def from_json(cls, body):
        return LaserParam(
            vec=np.array(body["vec"]),
            error=body["error"],
        )


@dataclass(slots=True)
class LaserParamProfile:
    name: str
    param: LaserParam

    def to_json(self):
        return {
            "name": self.name,
            "param": self.param.to_json(),
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            name=json_data["name"],
            param=LaserParam.from_json(json_data["param"]),
        )
