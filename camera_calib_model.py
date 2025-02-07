import time
import cv2
import numpy as np

import numpy as np


def rotation_matrix_from_vectors(v1, v2):
    """
    v1をv2にマッピングする回転行列を計算
    """
    # 正規化（単位ベクトルにする）
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 回転軸を求める
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # もし軸がゼロなら（v1とv2が同じ or 反対向き）
    if axis_norm < 1e-8:
        if np.dot(v1, v2) > 0:  # 同じ向きなら単位行列
            return np.eye(3)
        else:  # 反対向きなら適当な軸で180度回転
            return -np.eye(3)

    axis = axis / axis_norm  # 正規化

    # 回転角を求める
    cos_theta = np.dot(v1, v2)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    # Rodriguesの回転公式を適用
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    return R


def normal_from_three_points(p1, p2, p3, normalize=True):
    """
    3つの3D点から法線ベクトルを求める

    :param p1, p2, p3: 3つの点 (numpy array)
    :param normalize: Trueなら単位ベクトルに正規化
    :return: 法線ベクトル (numpy array)
    """
    v1 = p2 - p1  # 1つ目の辺のベクトル
    v2 = p3 - p1  # 2つ目の辺のベクトル

    normal = np.cross(v1, v2)  # 外積を計算

    if normalize:
        normal = normal / np.linalg.norm(normal)  # 単位ベクトル化

    return normal


class CameraCalibModel:
    def __init__(self, w_points: np.ndarray, planes: np.ndarray, p_dist: float, v_eye: np.ndarray,
                 v_light: np.ndarray, f: float):
        self._w_points = w_points  # ndim=2, (N, 3), unit: mm
        self._planes = planes  # ndim=2, (N, 4, 3), unit: mm
        self._p_dist = p_dist  # eye distance from origin
        self._v_eye = v_eye  # ndim=1, (3,), unit: mm  eye direction
        self._v_light = v_light  # ndim=1, (3,), unit: mm  light direction
        self._f = f  # focal length (float)

    def get_world_point_count(self) -> int:
        return self._w_points.shape[0]

    def get_world_point(self, i) -> tuple[float, float, float]:
        return tuple(map(float, self._w_points[i]))

    def render_3d(
            self,
            width: int,
            height: int,
            p_highlight: int | None = None,
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), np.uint8)

        v_eye = self._v_eye / np.linalg.norm(self._v_eye)
        v_eye = v_eye + np.array([
            0.05 * np.sin(2 * np.pi * 0.1 * time.time()),
            0.01 * np.cos(2 * np.pi * 0.1 * time.time()),
            0,
        ])
        p_eye = self._p_dist * -v_eye
        v_light = self._v_light / np.linalg.norm(self._v_light)
        f = self._f * min(width, height)

        mat_world_rot = np.eye(4)
        mat_world_rot[:3, :3] = rotation_matrix_from_vectors(v_eye, np.array([0, 0, 1]))

        mat_world_trans = np.array([
            [1, 0, 0, -p_eye[0]],
            [0, 1, 0, -p_eye[1]],
            [0, 0, 1, -p_eye[2]],
            [0, 0, 1, 0],
        ])

        mat_world_to_camera = mat_world_rot @ mat_world_trans

        mat_world_rot_correct = np.eye(4)
        mat_world_rot_correct[:3, :3] @= np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])

        mat_world_to_camera = mat_world_rot_correct @ mat_world_to_camera

        ofs_x, ofs_y = width / 2, height / 2

        def convert(p):
            x, y, z, _ = mat_world_to_camera @ np.array([*p, 1])
            return np.array([x / z * f + ofs_x, y / z * f + ofs_y])

        # draw planes
        for plane in self._planes:
            p1, p2, p3, p4 = plane

            n = normal_from_three_points(p1, p2, p3)
            brightness = abs(np.dot(n, v_light))
            brightness = brightness * 0.8 + 0.2
            color = tuple([int(255 * brightness)] * 3)

            p1, p2 = convert(p1), convert(p2)
            p3, p4 = convert(p3), convert(p4)

            cv2.fillConvexPoly(
                canvas,
                np.array([[p1], [p2], [p3]]).astype(np.int32),
                color,
            )

            cv2.fillConvexPoly(
                canvas,
                np.array([[p3], [p4], [p1]]).astype(np.int32),
                color,
            )

        # draw points
        for i, p in enumerate(self._w_points):
            if i == p_highlight:
                color = 0, 0, 255
                thickness = 3
            else:
                color = 255, 0, 0
                thickness = 1
            p2d = convert(p)
            if p2d is not None:
                cv2.circle(
                    canvas,
                    (int(p2d[0]), int(p2d[1])),
                    2,
                    color,
                    thickness,
                )

        return canvas


DEFAULT_CALIB_MODEL = CameraCalibModel(
    w_points=np.array([
        [0, 10, 70],
        [0, 30, 70],
        [0, 30, 50],
        [0, 50, 70],
        [0, 50, 50],
        [0, 50, 30],
        [0, 70, 70],
        [0, 70, 50],
        [0, 70, 30],
        [0, 70, 10],
        [10, 75, 0],
        [30, 75, 0],
        [50, 75, 0],
        [70, 75, 0],
        [10, 65, 0],
        [30, 65, 0],
        [50, 65, 0],
        [70, 65, 0],
        [10, 55, 20],
        [30, 55, 20],
        [50, 55, 20],
        [70, 55, 20],
        [10, 45, 20],
        [30, 45, 20],
        [50, 45, 20],
        [70, 45, 20],
        [10, 35, 40],
        [30, 35, 40],
        [50, 35, 40],
        [70, 35, 40],
        [10, 25, 40],
        [30, 25, 40],
        [50, 25, 40],
        [70, 25, 40],
        [10, 15, 60],
        [30, 15, 60],
        [50, 15, 60],
        [70, 15, 60],
        [10, 5, 60],
        [30, 5, 60],
        [50, 5, 60],
        [70, 5, 60],
        [10, 60, 5],
        [30, 60, 5],
        [50, 60, 5],
        [70, 60, 5],
        [10, 60, 15],
        [30, 60, 15],
        [50, 60, 15],
        [70, 60, 15],
        [10, 40, 25],
        [30, 40, 25],
        [50, 40, 25],
        [70, 40, 25],
        [10, 40, 35],
        [30, 40, 35],
        [50, 40, 35],
        [70, 40, 35],
        [10, 20, 45],
        [30, 20, 45],
        [50, 20, 45],
        [70, 20, 45],
        [10, 20, 55],
        [30, 20, 55],
        [50, 20, 55],
        [70, 20, 55],
        [10, 0, 65],
        [30, 0, 65],
        [50, 0, 65],
        [70, 0, 65],
        [10, 0, 75],
        [30, 0, 75],
        [50, 0, 75],
        [70, 0, 75],
    ]),
    planes=np.array([
        [
            [0, 0, 0],
            [0, 80, 0],
            [0, 80, 80],
            [0, 0, 80],
        ],
        [
            [0, 0, 0],
            [80, 0, 0],
            [80, 80, 0],
            [0, 80, 0],
        ],
        [
            [0, 0, 0],
            [80, 0, 0],
            [80, 0, 80],
            [0, 0, 80],
        ],
        *(
            [
                [0, 20 * i + 20, 60 - 20 * i],
                [80, 20 * i + 20, 60 - 20 * i],
                [80, 20 * i, 60 - 20 * i],
                [0, 20 * i, 60 - 20 * i],
            ] for i in range(4)
        ),
        *(
            [
                [0, 60 - i * 20, 20 * i],
                [80, 60 - i * 20, 20 * i],
                [80, 60 - i * 20, 20 * i + 20],
                [0, 60 - i * 20, 20 * i + 20],
            ] for i in range(4)
        ),
    ]),
    p_dist=300,
    v_eye=np.array([-0.9, -0.9, -1]),
    v_light=np.array([-0.3, -0.4, -0.6]),
    f=1.3,
)

# while True:
#     im = DEFAULT_CALIB_MODEL.render_3d(500, 500, p_highlight=int(time.time() * 3) % 60)
#
#     cv2.imshow("win", im)
#     if cv2.waitKey(1) == ord("q"):
#         break
