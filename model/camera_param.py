from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraParamProfile:
    name: str
    mat: np.ndarray

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
