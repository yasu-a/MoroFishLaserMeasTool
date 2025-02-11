from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane3D:
    # crate plane of ax + by + cz + 1 = 0
    a: float
    b: float
    c: float

    @classmethod
    def from_points(cls, points: np.ndarray, is_complete_plane=True, return_error=False):
        points = np.array(points, np.float64)
        assert points.ndim == 2 and points.shape[1] == 3, \
            f"Points have invalid shape {points.shape}"

        a, b, c, = np.linalg.pinv(points) @ np.full(len(points), -1, np.float64)
        error = np.sum(np.square(np.array([a, b, c]) @ points.T + 1))
        if is_complete_plane and error > 1e-6:
            for i in range(3):
                if np.all(points[:, i] <= 1e-10):
                    # z=0やy=0のような平面を ax + by + cz + 1 = 0 で表すと係数が無限大になるので補正
                    points = points.copy()
                    points[:, i] += 1e-10
                    return cls.from_points(points)
            else:
                raise ValueError("Points do not form a plane")

        p = cls(a, b, c)
        if return_error:
            return p, error
        else:
            return p
