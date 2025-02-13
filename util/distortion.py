import pickle

import cv2
import numpy as np


class DistortionCorrector:
    def __init__(self):
        self._ret = None
        self._mtx = None
        self._dist = None
        self._rvecs = None
        self._tvecs = None

        self._map: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}  # size -> maps

    def __repr__(self):
        return f"Params(ret={self._ret}, mtx={self._mtx}, dist={self._dist}, rvecs={self._rvecs}, tvecs={self._tvecs})"

    def get_map(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        size = width, height
        if size not in self._map:
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self._mtx,
                self._dist,
                (width, height),
                1,
                (width, height),
            )
            self._map[size] = cv2.initUndistortRectifyMap(
                self._mtx,
                self._dist,
                None,
                new_mtx,
                (width, height),
                cv2.CV_16SC2,
            )
        return self._map[size]

    def update_map(self) -> None:
        self._map.clear()

    def load(self, path) -> None:
        with open(path, 'rb') as f:
            self._ret, self._mtx, self._dist, self._rvecs, self._tvecs = pickle.load(f)
        self.update_map()

    def set_new_param(self, ret, mtx, dist, rvecs, tvecs) -> None:
        self._ret, self._mtx, self._dist, self._rvecs, self._tvecs = ret, mtx, dist, rvecs, tvecs
        self.update_map()

    def undistort(self, im: np.ndarray) -> np.ndarray:
        height, width = im.shape[:2]
        map1, map2 = self.get_map(width, height)
        return cv2.remap(im, map1, map2, cv2.INTER_LINEAR)

    def is_valid(self) -> bool:
        return self._ret is not None

    def calibrate(self, objpoints, imgpoints, width, height):
        ret, mtx, dist, rvecs, tvecs \
            = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        self.set_new_param(ret, mtx, dist, rvecs, tvecs)
        return ret, mtx, dist, rvecs, tvecs

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self._ret, self._mtx, self._dist, self._rvecs, self._tvecs), f)
