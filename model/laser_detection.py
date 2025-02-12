from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class LaserDetectionModel:
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    morph_open_size: int
    morph_close_size: int

    def __post_init__(self):
        assert isinstance(self.hsv_min, np.ndarray), "hsv_min should be a numpy array"
        assert self.hsv_min.ndim == 1, "hsv_min should be 1D"
        assert self.hsv_min.shape == (3,), "hsv_min should have shape of (3,)"

        assert isinstance(self.hsv_max, np.ndarray), "hsv_max should be a numpy array"
        assert self.hsv_max.ndim == 1, "hsv_max should be 1D"
        assert self.hsv_max.shape == (3,), "hsv_max should have shape of (3,)"

    def to_json(self):
        return {
            "hsv_min": self.hsv_min.tolist(),
            "hsv_max": self.hsv_max.tolist(),
            "morph_open_size": self.morph_open_size,
            "morph_close_size": self.morph_close_size,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            hsv_min=np.array(body["hsv_min"]),
            hsv_max=np.array(body["hsv_max"]),
            morph_open_size=body["morph_open_size"],
            morph_close_size=body["morph_close_size"],
        )

    _MORPH_KERNEL = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], np.uint8)

    def create_laser_mask(self, im: np.ndarray, is_hsv=False):  # TODO: move to service
        if not is_hsv:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(im, self.hsv_min, self.hsv_max)
        if self.morph_open_size > 0:
            cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._MORPH_KERNEL,
                iterations=self.morph_open_size,
                dst=mask,
            )
        if self.morph_close_size > 0:
            cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._MORPH_KERNEL,
                iterations=self.morph_close_size,
                dst=mask,
            )
        return mask


@dataclass(slots=True)
class LaserDetectionProfile:
    name: str
    model: LaserDetectionModel

    def to_json(self):
        return {
            "name": self.name,
            "param": self.model.to_json(),
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            name=body["name"],
            model=LaserDetectionModel.from_json(body["param"]),
        )
