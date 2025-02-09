from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DistortionParameters:
    ret: float
    mtx: np.ndarray
    dist: np.ndarray
    rvecs: tuple[np.ndarray]
    tvecs: tuple[np.ndarray]

    def undistort(self, im: np.ndarray):  # TODO: move to service
        h, w = im.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h)
        )
        im_undistort = cv2.undistort(im, self.mtx, self.dist, None, new_mtx)
        return im_undistort


@dataclass(slots=True)
class DistortionProfile:
    name: str
    params: DistortionParameters
