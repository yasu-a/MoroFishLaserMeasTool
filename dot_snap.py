import functools

import cv2
import numpy as np
from matplotlib import pyplot as plt

from app_logging import create_logger


class DotSnapComputerError(RuntimeError):
    pass


class DotSnapComputer:
    _logger = create_logger()

    def __init__(
            self,
            im_gray: np.ndarray,
            *,
            crop_radius: int,
            min_samples: int,
            snap_radius: float,
            stride: int,
    ):
        self._im_gray = im_gray  # スナップ点を探す画像（グレースケール）
        self._crop_radius = crop_radius  # スナップ点を探索するときに画像から切り出す小矩形領域の内接円半径
        self._min_samples = min_samples  # 全体のスナップ点を集計して座標クラスタリングしたときのスナップの対象とする点が所属する最小クラスタサイズ
        self._snap_radius = snap_radius  # スナップを効かせるときのクエリ点とスナップ点の最大距離
        self._stride = stride  # 全体のスナップ点を計算するときに何ピクセルおきに計算するか

        self._snap_points: np.ndarray | None = None  # snap points [[x, y], ...]

    @staticmethod
    def _crop_image(im_gray: np.ndarray, center: tuple[int, int], crop_radius: int):
        im_gray_pad = np.pad(
            im_gray,
            pad_width=((crop_radius, crop_radius), (crop_radius, crop_radius)),
            mode="median",
        )
        x_begin = int(crop_radius + center[0] - crop_radius)
        x_end = int(crop_radius + center[0] + 1 + crop_radius)
        y_begin = int(crop_radius + center[1] - crop_radius)
        y_end = int(crop_radius + center[1] + 1 + crop_radius)
        im_gray_crop = im_gray_pad[y_begin:y_end, x_begin:x_end]
        return im_gray_crop

    @staticmethod
    def _has_low_contrast(im_gray: np.ndarray) -> bool:
        return im_gray.std() <= 2

    @staticmethod
    def _classify_pixels(im_gray: np.ndarray) -> np.ndarray:
        # 大津の二値化による{0, 1}への分類
        _, im_cls = cv2.threshold(im_gray, 0, 1, cv2.THRESH_OTSU)
        return im_cls

    @staticmethod
    def _count_connected_components(im_gray: np.ndarray) -> int:
        n_connected_components, _ = cv2.connectedComponents(im_gray)
        return n_connected_components

    @staticmethod
    def _create_pixel_points_of(height: int, width: int) -> np.ndarray:
        return np.indices((height, width)).transpose((2, 1, 0))

    @staticmethod
    def _get_centroid_and_std(pixel_points: np.ndarray, mask: np.ndarray) \
            -> tuple[np.ndarray, float]:
        points = pixel_points[np.bool_(mask), :]
        return points.mean(axis=0), np.linalg.norm(points.std(axis=0))

    @staticmethod
    def _mask_borders(arr, num=1):
        # https://stackoverflow.com/questions/41200719/how-to-get-all-array-edges
        mask = np.zeros(arr.shape, bool)
        for dim in range(arr.ndim):
            mask[tuple(
                slice(0, num) if idx == dim else slice(None) for idx in range(arr.ndim))] = True
            mask[tuple(slice(-num, None) if idx == dim else slice(None) for idx in
                       range(arr.ndim))] = True
        return mask

    def compute_snap_positions(self) -> np.ndarray:  # [[x, y], ...]: int
        # 適応的二値化
        im_bin = cv2.adaptiveThreshold(
            self._im_gray,
            255,  # maxValue
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,  # blockSize
            4,  # C
        )

        # モルフォロジー変換（ノイズ除去）
        im_bin_morph = cv2.morphologyEx(
            im_bin,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=max(1, int(round(self._im_gray.shape[1] / 1900 * 3, 0))),
        )

        # 輪郭を計算
        im_bin_morph[self._mask_borders(im_bin_morph)] = im_bin_morph.max()  # 端の点の輪郭をとるため
        contours, _ = cv2.findContours(im_bin_morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # スタットをとる
        ratios = []
        centers_xy = []
        areas = []
        for c in contours:
            try:
                (x, y), (major, minor), angle = cv2.fitEllipse(c)
            except cv2.error:
                continue
            ratios.append(min(major / (minor + 1e-6), minor / (major + 1e-6)))
            centers_xy.append([x, y])
            areas.append(np.pi * major * minor)
        ratios = np.array(ratios)
        centers_xy = np.array(centers_xy).round(0).astype(int)
        areas = np.array(areas)

        # 抽出条件
        cond = (ratios > 0.4) & (5 < areas) & (areas < 60 * 60 * np.pi)

        # 点抽出
        centers_xy = centers_xy[cond]
        centers_xy = np.unique(centers_xy, axis=0)  # 重複点削除

        return centers_xy

    def _get_snap_points(self) -> np.ndarray:
        if self._snap_points is None:
            self._logger.info("Computing snap points...")
            self._snap_points = self.compute_snap_positions()
            self._logger.info(f"Found {len(self._snap_points)} points:\n{self._snap_points}")
            self._logger.info("Computing snap points finished")
        return self._snap_points

    @functools.cache
    def find_snap_pos(self, current_pos: tuple[int, int]) -> tuple[int, int] | None:
        snap_points = self._get_snap_points()
        distances = np.linalg.norm(snap_points - current_pos, axis=1)
        index = np.arange(len(snap_points))[distances <= self._snap_radius]
        if len(index) == 0:
            return None
        snap_x, snap_y = snap_points[index[np.argmin(distances[index])]]
        return int(snap_x), int(snap_y)
