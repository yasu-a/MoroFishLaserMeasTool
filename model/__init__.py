from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DistortionParameters:
    ret: float
    mtx: np.ndarray
    dist: np.ndarray
    rvecs: tuple[np.ndarray]
    tvecs: tuple[np.ndarray]


@dataclass(slots=True)
class DistortionCorrectionProfile:
    name: str
    params: DistortionParameters


@dataclass(slots=True)
class Image:
    name: str
    data: np.ndarray
