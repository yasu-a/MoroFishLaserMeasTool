from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LaserDetectionProfile:
    name: str
    hsv_mean: np.ndarray
    hsv_cov: np.ndarray

    def __post_init__(self):
        assert isinstance(self.hsv_mean, np.ndarray), "hsv_mean should be a numpy array"
        assert self.hsv_mean.ndim == 1, "hsv_mean should be 1D"
        assert self.hsv_mean.shape == (3,), "hsv_mean should have shape of (3,)"
        assert isinstance(self.hsv_cov, np.ndarray), "hsv_cov should be a numpy array"
        assert self.hsv_cov.ndim == 2, "hsv_cov should be 2D"
        assert self.hsv_cov.shape == (3, 3), "hsv_cov should have shape of (3, 3)"

    def to_json(self):
        return {
            "name": self.name,
            "hsv_mean": self.hsv_mean.tolist(),
            "hsv_cov": self.hsv_cov.tolist()
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            name=body["name"],
            hsv_mean=np.array(body["hsv_mean"]),
            hsv_cov=np.array(body["hsv_cov"])
        )
