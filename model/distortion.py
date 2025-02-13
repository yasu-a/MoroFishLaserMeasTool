from dataclasses import dataclass
from functools import cached_property

import numpy as np

from util.distortion import DistortionCorrector


@dataclass
class DistortionParameters:
    ret: float
    mtx: np.ndarray
    dist: np.ndarray
    rvecs: tuple[np.ndarray]
    tvecs: tuple[np.ndarray]

    @cached_property
    def _corrector(self) -> DistortionCorrector:
        c = DistortionCorrector()
        c.set_new_param(
            ret=self.ret,
            mtx=self.mtx,
            dist=self.dist,
            rvecs=self.rvecs,
            tvecs=self.tvecs,
        )
        return c

    def undistort(self, im: np.ndarray):  # TODO: move to service
        return self._corrector.undistort(im)


@dataclass(slots=True)
class DistortionProfile:
    name: str
    params: DistortionParameters
