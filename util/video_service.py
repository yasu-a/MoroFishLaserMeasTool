import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import repo.video
from app_logging import create_logger
from model.camera_param import CameraParam
from model.global_config import ROI
from model.laser_detection import LaserDetectionModel
from model.laser_param import LaserParam
from model.video import Video
from util.light_section_method import get_laser_2d_and_3d_points
from video_reader import VideoReader

_logger = create_logger()


def video_fullpath(video: Video) -> Path:
    return repo.video.get_item_path(video.name, "video.mp4")


def create_video_from_file(src_video_path: Path, video_name: str) -> Video:
    assert src_video_path.suffix == ".mp4", src_video_path
    video = Video(name=video_name)
    repo.video.put(video)
    dst_video_path = video_fullpath(video)
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)
    src_video_path.rename(dst_video_path)
    return video


def get_video_reader(video: Video) -> VideoReader:
    video_path = repo.video.get_item_path(video.name, "video.mp4")
    return VideoReader(str(video_path))


def frames_fullpath(video: Video) -> Path:
    return repo.video.get_item_path(video.name, "frames.h5")


def fg_frames_fullpath(video: Video) -> Path:
    return repo.video.get_item_path(video.name, "fg_frames.h5")


def stitching_frames_fullpath(video: Video) -> Path:
    return repo.video.get_item_path(video.name, "stitching_frames.pickle")


def save_image(im, video, name):
    path = repo.video.get_item_path(video.name, f"_{name}.png")
    cv2.imwrite(str(path), im)


def clear_dynamic_data(video: Video, fg=True, stitching_frames=True) -> None:
    paths = []
    if fg:
        paths.extend([
            frames_fullpath(video),
            fg_frames_fullpath(video),
        ])
        stitching_frames = True
    if stitching_frames:
        paths.extend([
            stitching_frames_fullpath(video),
        ])

    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


def get_video_shape(video: Video) -> tuple[int, int, int]:  # n_frames, height, width
    cap = cv2.VideoCapture(str(video_fullpath(video)))
    if not cap.isOpened():
        raise ValueError(
            f"Failed to open video file: {video_fullpath(video)}"
        )
    try:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n_frames, height, width
    finally:
        cap.release()


def create_frames_roi(video: Video, roi: ROI) -> None:
    if frames_fullpath(video).exists():
        return

    _logger.info(f"Creating frames: {video.name}, ROI(y,x)={roi.get_image_slice()}")

    n_frames, height, width = get_video_shape(video)

    cap = cv2.VideoCapture(str(video_fullpath(video)))
    if not cap.isOpened():
        raise ValueError(
            f"Failed to open video file: {video_fullpath(video)}"
        )

    try:
        roi_height = min(height, roi.screen_y_max) - max(0, roi.screen_y_min)
        roi_width = min(width, roi.screen_x_max) - max(0, roi.screen_x_min)
        with h5py.File(frames_fullpath(video), "w") as f_frames:
            f_frames.create_dataset(
                "data",
                shape=(n_frames, roi_height, roi_width, 3),
                dtype=np.uint8,
            )
            i = 0
            while True:
                flag, frame = cap.read()
                if not flag:
                    break
                f_frames["data"][i, :, :, :] = frame[roi.get_image_slice()]
                i += 1
    finally:
        cap.release()


@dataclass(frozen=True)
class StitchConfig:
    fg_w_edge: int
    fg_feature_q_large: int
    fg_feature_q_small: int
    fg_n_end: int
    fg_n_elim: int
    fg_feature_eps: float
    fg_feature_min_pts: int
    fg_cov_lambda: float
    fg_th_mahalanobis: int
    fg_mask_pp_morph_open_px: int
    fg_mask_pp_morph_close_px: int
    tm_max_move: tuple[int, int]
    tm_min_move: tuple[int, int]
    tm_n_horizontal_split: int
    stitch_th_fg_overlap: int


def get_frame_feature(video: Video, config: StitchConfig) -> np.ndarray:
    with h5py.File(frames_fullpath(video)) as f_frames:
        frames = f_frames["data"]

        frames_edge = np.concatenate([
            frames[:, :, :config.fg_w_edge, :],
            frames[:, :, -config.fg_w_edge:, :],
        ], axis=-2)  # (N, H, W_EDGE * 2, 3)

        # 最大最小
        q_large, q_small = np.percentile(
            frames_edge,
            q=[config.fg_feature_q_small, config.fg_feature_q_large],
            axis=(1, 2),
        )
        features = np.concatenate([
            q_large,
            q_small,
        ], axis=-1)  # (N, 6)

        features = features / 255  # N, 6
        assert len(features) == len(frames)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

        return features


def get_background_frame_indexes(video: Video, config: StitchConfig) -> np.ndarray:
    features = get_frame_feature(video, config)

    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=config.fg_feature_eps, min_samples=config.fg_feature_min_pts)
    labels = dbscan.fit_predict(features)

    bg_ensured_frame_indexes = np.concatenate([
        np.array([
            i
            for i in range(config.fg_n_end)
            if i > config.fg_n_elim
        ]),
        np.array([
            i
            for i in range(len(features) - config.fg_n_end, len(features))
            if i < len(features) - config.fg_n_elim
        ]),
    ]).astype(np.int32)

    labels_bg_ensured = np.array(list(set(labels[bg_ensured_frame_indexes]) - {-1}))
    fg_flags = np.in1d(labels, labels_bg_ensured)
    fg_indexes = np.where(fg_flags)[0]

    _logger.info(f"Background frame indexes: {fg_indexes}")

    return fg_indexes


def get_background_model(video: Video, config: StitchConfig) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # mean_est, cov_est, inv_cov_est
    def _calculate_cov(frames):
        assert frames.ndim == 4 and frames.shape[-1] == 3, frames.shape
        # frames の形状: (フレーム数, 高さ, 幅, 3)
        nf, h, w, _ = frames.shape

        # 平均を計算: (高さ, 幅, 3)
        mean = np.mean(frames, axis=0)

        # フレーム方向に沿ってピクセルごとの偏差を計算: (フレーム数, 高さ, 幅, 3)
        deviations = frames - mean

        # 共分散行列を計算: (高さ * 幅, 3, 3)
        # noinspection SpellCheckingInspection
        cov_matrices = np.einsum(
            'fxyi,fxyj->xyij',
            deviations,
            deviations,
        ) / (nf - 1)

        return cov_matrices

    # FIXME: 近似のせいでマハラノビス距離がマイナスになることがある
    # def _calculate_fast_cov(frames, scale=0.33):
    #     assert frames.ndim == 4 and frames.shape[-1] == 3, frames.shape
    #
    #     _, h, w, _ = frames.shape
    #     frames_scaled = np.array([
    #         cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    #         for frame in frames
    #     ])
    #     cov_scaled = _calculate_cov(frames_scaled)
    #     cov = np.zeros_like(cov_scaled, shape=(h, w, 3, 3))
    #     for i in range(3):
    #         for j in range(3):
    #             cov[:, :, i, j] = cv2.resize(cov_scaled[:, :, i, j], (w, h),
    #                                          interpolation=cv2.INTER_CUBIC)
    #     return cov

    def create_model():
        bg_indexes = get_background_frame_indexes(video, config)

        with h5py.File(frames_fullpath(video)) as f_frames:
            frames = f_frames["data"]

            frames_target = frames[bg_indexes, :, :, :]
            mean_est = np.mean(frames_target, axis=0)
            cov_est = _calculate_cov(frames_target) \
                      + (np.eye(3) * config.fg_cov_lambda)[None, None, :, :]
            inv_cov_est = np.linalg.inv(cov_est)

        return mean_est, cov_est, inv_cov_est

    return create_model()


def create_masked_preview(im, mask):
    im = im.copy()
    im[mask.astype(bool)] = cv2.addWeighted(
        im,
        1,
        np.dstack([
            np.zeros_like(mask[:, :]),
            np.zeros_like(mask[:, :]),
            np.full_like(mask[:, :], fill_value=255),
        ]),
        0.5,
        0,
    )[mask.astype(bool)]
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        im,
        contours,
        -1,
        (0, 0, 255),
        3,
    )
    return im


def create_foreground(video: Video, config: StitchConfig) -> None:
    if fg_frames_fullpath(video).exists():
        return

    _logger.info(f"Creating foreground: {video.name}")

    PP_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def _post_process_mask(mask, preview_with_frame=None):
        # def preview_if_needed(mask):
        #     if preview_with_frame is None:
        #         return
        #     im = preview_with_frame.copy()
        #     red_mask = np.dstack([
        #         np.zeros_like(mask),
        #         np.zeros_like(mask),
        #         mask,
        #     ])
        #     im = cv2.addWeighted(
        #         im,
        #         1,
        #         red_mask,
        #         0.5,
        #         0,
        #     )
        #     contours, _ = cv2.findContours(
        #         mask,
        #         cv2.RETR_LIST,
        #         cv2.CHAIN_APPROX_SIMPLE,
        #     )
        #     cv2.drawContours(
        #         im,
        #         contours,
        #         -1,
        #         (0, 0, 255),
        #         3,
        #     )
        #     im = cv2.resize(im, None, fx=0.5, fy=0.5)
        #     cv2_imshow(im)

        # preview_if_needed(mask)

        mask = cv2.morphologyEx(  # 点をつぶしてから
            mask,
            cv2.MORPH_OPEN,
            PP_KERNEL,
            None,
            iterations=config.fg_mask_pp_morph_open_px,
        )

        # preview_if_needed(mask)

        mask = cv2.morphologyEx(  # 穴をつぶす
            mask,
            cv2.MORPH_CLOSE,
            PP_KERNEL,
            None,
            iterations=config.fg_mask_pp_morph_close_px,
        )

        # preview_if_needed(mask)

        # get area with the largest external contour
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        mask[:, :] = 0
        if contours:
            i_max = np.argmax([cv2.contourArea(c) for c in contours])
            # noinspection PyTypeChecker
            mask = cv2.drawContours(
                mask,
                contours,
                i_max,
                255,
                -1,
            )

        # preview_if_needed(mask)

        return mask

    mean_est, cov_est, inv_cov_est = get_background_model(video, config)

    with h5py.File(frames_fullpath(video), "r") as f_frames:
        frames = f_frames["data"]
        n_frames = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]

        with h5py.File(fg_frames_fullpath(video), "w") as f_fg_frames:
            f_fg_frames.create_dataset(
                "data",
                shape=(n_frames, height, width),
                dtype=np.uint8,
            )

            for i in tqdm(range(n_frames)):
                frame = frames[i]
                deviations = frame - mean_est
                distances_squared = np.einsum(
                    'ijk,ijkl,ijl->ij', deviations, inv_cov_est, deviations
                )
                mask = np.where(
                    distances_squared >= config.fg_th_mahalanobis ** 2, np.uint8(255), np.uint8(0)
                )
                mask_processed = _post_process_mask(mask)
                f_fg_frames["data"][i, :, :] = mask_processed

                cv2.imshow("fg", create_masked_preview(frame, mask_processed))
                # cv2.imshow(
                #     "fg",
                #     cv2.normalize(
                #         np.sqrt(distances_squared),
                #         None,
                #         0,
                #         255,
                #         cv2.NORM_MINMAX,
                #         cv2.CV_8U,
                #     )
                # )
                cv2.waitKey(1)

            cv2.destroyWindow("fg")


@dataclass(slots=True)
class StitchingFrame:
    im: np.ndarray
    fg_mask: np.ndarray
    laser_mask: np.ndarray
    points_screen: np.ndarray
    points_world: np.ndarray

    @property
    def im_tm(self) -> np.ndarray:
        im = self.im.copy()

        im = cv2.GaussianBlur(im.astype(float), (0, 0), 1, cv2.CV_32F)

        im = cv2.Laplacian(
            im,
            cv2.CV_64F,
            ksize=3,
        )

        mean = im.mean()
        std = im.std()

        im = (im - mean) / (std * 2)

        im = cv2.normalize(
            im,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )

        return im

    @property
    def width(self) -> int:
        return self.im.shape[1]

    @property
    def height(self) -> int:
        return self.im.shape[0]

    @property
    def has_laser(self) -> bool:
        return len(self.points_screen) > 0

    @property
    def fg_ratio(self) -> float:
        vertical_count = np.count_nonzero(self.fg_mask, axis=0)
        vertical_ratio = vertical_count / self.height
        vertical_flag = vertical_ratio > 0.03
        horizontal_count = np.count_nonzero(vertical_flag)
        horizontal_ratio = horizontal_count / self.width
        return horizontal_ratio


def get_stitching_frames(
        video: Video,
        roi: ROI,
        camera_param: CameraParam,
        laser_param: LaserParam,
        laser_detection_model: LaserDetectionModel,
        n_mid_frames: int,
) -> list[StitchingFrame]:
    _, src_height, src_width = get_video_shape(video)

    def iter_frames() -> Iterable[StitchingFrame]:
        with h5py.File(frames_fullpath(video), "r") as f_frames:
            with h5py.File(fg_frames_fullpath(video), "r") as f_fg_frames:
                frames = f_frames["data"]
                fg_frames = f_fg_frames["data"]

                _, src_height, src_width = get_video_shape(video)

                for i in range(len(frames)):
                    im = frames[i, :, :, :]
                    fg_mask = fg_frames[i, :, :]

                    laser_det_mask = laser_detection_model.create_laser_mask(im)
                    laser_mask = np.zeros((src_height, src_width), np.uint8)
                    s = (
                        slice(roi.screen_y_min, roi.screen_y_min + laser_det_mask.shape[0]),
                        slice(roi.screen_x_min, roi.screen_x_min + laser_det_mask.shape[1]),
                    )
                    laser_mask[s] = laser_det_mask
                    points_screen, points_world = get_laser_2d_and_3d_points(
                        laser_mask,
                        camera_param,
                        laser_param,
                        roi,
                    )
                    points_screen[:, 0] -= roi.screen_x_min
                    points_screen[:, 1] -= roi.screen_y_min

                    yield StitchingFrame(
                        im=im,
                        fg_mask=fg_mask,
                        laser_mask=laser_det_mask,
                        points_screen=points_screen,
                        points_world=points_world,
                    )

    if stitching_frames_fullpath(video).exists():
        with stitching_frames_fullpath(video).open("rb") as f:
            lst = pickle.load(f)
            assert isinstance(lst, list) and all(isinstance(item, StitchingFrame) for item in lst)
    else:
        lst = []
        state = 0
        mid_frames = []
        for f in iter_frames():
            if state == 0:
                if f.has_laser and f.fg_ratio >= 0.9:
                    lst.append(f)
                    state = 1
            elif state == 1:
                if f.fg_ratio >= 0.975:
                    state = 2
            elif state == 2:
                if f.has_laser and f.fg_ratio >= 0.99:
                    state = 3
                    mid_frames.append(f)
            elif state == 3:
                mid_frames.append(f)
                if f.fg_ratio <= 0.975:
                    for k in range(1, n_mid_frames + 1):
                        lst.append(mid_frames[len(mid_frames) // (n_mid_frames + 1) * k])
                    state = 4
            elif state == 4:
                if f.has_laser and f.fg_ratio <= 0.9:
                    lst.append(f)
                    state = 99
            elif state == 99:
                break

            im = create_masked_preview(f.im, f.fg_mask)
            last_v = 0
            for i in range(len(f.points_screen)):
                u, v = f.points_screen[i]
                x, y, z = f.points_world[i]
                cv2.circle(
                    im,
                    (int(u), int(v)),
                    2,
                    (0, 0, 255),
                    1,
                )
                if abs(last_v - v) >= 20:
                    cv2.putText(
                        im,
                        f"({x:.0f}, {y:.0f}, {z:.0f})",
                        (int(u), int(v) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    last_v = v
            cv2.putText(
                im,
                f"State: {state}, fg: {f.fg_ratio}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("mask_and_laser", im)
            if lst:
                cv2.imshow(
                    "stitching_frames",
                    np.hstack([cv2.resize(a.im, None, fx=0.3, fy=0.3) for a in lst]),
                )
            cv2.waitKey(1)

        cv2.waitKey(1000)
        cv2.destroyWindow("mask_and_laser")
        cv2.destroyWindow("stitching_frames")

        save_image(np.hstack([a.im for a in lst]), video, "stitching_frames")
        with stitching_frames_fullpath(video).open("wb") as f:
            pickle.dump(lst, f)

    return lst


def template_match(f_1: StitchingFrame, f_2: StitchingFrame, config: StitchConfig) \
        -> tuple[int, int]:  # matching image and (delta x, delta y)

    mx, my = config.tm_max_move

    results = []
    for k in range(config.tm_n_horizontal_split):
        im_1 = f_1.im_tm.copy()
        mask_1 = f_1.fg_mask.copy()
        mask_laser_1 = f_1.laser_mask.copy()

        im_2 = f_2.im_tm.copy()
        mask_2 = f_2.fg_mask.copy()

        h, w = im_1.shape[:2]
        w_split = w // config.tm_n_horizontal_split
        x_ofs = w_split * k

        im_1 = im_1[:, x_ofs:x_ofs + w_split]
        mask_1 = mask_1[:, x_ofs:x_ofs + w_split]
        mask_laser_1 = mask_laser_1[:, x_ofs:x_ofs + w_split]
        im_1[~np.bool_(mask_1)] = 0
        templ_mask = 255 - mask_laser_1.copy()

        im_2 = im_2.copy()
        im_2[~np.bool_(mask_2)] = 0
        im_2 = np.pad(
            im_2,
            ((my, my), (mx, mx), (0, 0)),
        )

        im_match = cv2.matchTemplate(
            image=im_2,
            templ=im_1,
            method=cv2.TM_SQDIFF_NORMED,
            mask=templ_mask,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(im_match)

        dx_global, dy_global = min_loc
        dx_local, dy_local = dx_global - mx - x_ofs, dy_global - my

        im = im_2.copy()
        im_paste = cv2.warpAffine(
            create_masked_preview(im_1, mask_1),
            np.float32([
                [1, 0, dx_global],
                [0, 1, dy_global],
            ]),
            (im.shape[1], im.shape[0]),
        )
        im = cv2.addWeighted(
            im,
            0.5,
            im_paste,
            0.5,
            0,
        )
        cv2.putText(
            im,
            f"{min_val:.2f} ({dx_local}, {dy_local})",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        results.append(
            dict(
                im=im,
                dx=dx_local,
                dy=dy_local,
                score=min_val,
            )
        )

    im_result = np.vstack([r["im"] for r in results])
    cv2.imshow("match", cv2.resize(im_result, None, fx=0.5, fy=0.5))
    cv2.waitKey(500)

    i_best = np.argmin([r["score"] for r in results])
    dx, dy = results[i_best]["dx"], results[i_best]["dy"]
    return dx, dy


def get_stitched_image(video: Video, config: StitchConfig) \
        -> tuple[np.ndarray, np.ndarray]:  # im and mask

    def _list_frame_index_and_trans(trans_lst: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
        mat = np.eye(3).astype(np.float32)
        lst: list[tuple[int, np.ndarray]] = []
        for i in range(len(trans_lst)):
            lst.append((i, mat.copy()))
            mat @= trans_lst[i]
        lst.append((len(trans_lst), mat.copy()))
        return lst

    def _calculate_stitch_geometry(w: int, h: int, trans_lst: list[np.ndarray], margin: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:  # offset, size
        corner_points_first = np.array([
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1],
        ], np.float32)

        corner_points_lst = []
        for i, mat in _list_frame_index_and_trans(trans_lst):
            corner_points_cur = cv2.perspectiveTransform(
                np.array([corner_points_first]),
                mat,
            )[0]
            corner_points_lst.append(corner_points_cur)
        corner_points_lst = np.concatenate(corner_points_lst)

        x, y, w, h = cv2.boundingRect(corner_points_lst)
        top_left = np.array([x, y])
        bottom_right = np.array([x + w, y + h])

        offset = (top_left - margin).round(0).astype(int)
        size = (bottom_right - top_left + margin * 2).round(0).astype(int)
        return offset, size  # [x, y], [x, y]

    _laser_close_morph_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def _stitch_frames(w: int, h: int, frames: np.ndarray, fg_masks: np.ndarray,
                       trans_lst: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate range of stitched image
        margin = np.array([32, 32])
        offset, size = _calculate_stitch_geometry(w, h, trans_lst, margin)

        # prepare stitching
        im_stitch = np.zeros(
            shape=(size[1], size[0], 3),
            dtype=np.uint16,
        )
        im_stitch_cur = im_stitch.copy()
        im_stitch_count = np.zeros(
            shape=(size[1], size[0]),
            dtype=np.uint8,
        )
        im_stitch_count_cur = im_stitch_count.copy()

        def construct_fg_mask_count_image():
            return im_stitch_count.copy()

        def construct_fg_mask_image():
            im = np.where(
                im_stitch_count >= config.stitch_th_fg_overlap,
                np.uint8(255),
                np.uint8(0),
            )
            # # レーザー部分を除去して合成すると切れることがあるのでモルフォロジーで埋める
            # if config.stitch_laser_elim_radius_px > 0:
            #     im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, _laser_close_morph_kernel,
            #                           iterations=config.stitch_laser_elim_radius_px * 2)
            return im

        def construct_frame_image():
            # 平均値をとる
            im_stitch_f = im_stitch.astype(np.float32)
            mask_lenient = im_stitch_count >= 1
            im_stitch_f[mask_lenient] /= im_stitch_count[mask_lenient, None]
            # マスク部分を除去
            mask_strict = construct_fg_mask_image()
            im_stitch_f[~mask_strict.astype(bool)] = 0
            return im_stitch_f.astype(np.uint8)

        # execute stitching
        for i, mat in tqdm(_list_frame_index_and_trans(trans_lst)):
            mat_stitch = mat.copy()
            mat_stitch[0, 2] -= offset[0]
            mat_stitch[1, 2] -= offset[1]

            im_frame = frames[i].copy()
            im_mask = fg_masks[i].copy()

            im_stitch_cur = cv2.warpPerspective(
                im_frame,
                mat_stitch,
                (size[0], size[1]),
                dst=im_stitch_cur,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            im_stitch_count_cur = cv2.warpPerspective(
                im_mask,
                mat_stitch,
                (size[0], size[1]),
                dst=im_stitch_count_cur,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            mask = im_stitch_count_cur >= 255
            im_stitch[mask] += im_stitch_cur[mask]
            im_stitch_count += mask

            print(i, mat.tolist())
            cv2.imshow("st", cv2.resize(construct_frame_image(), None, fx=0.5, fy=0.5))
            cv2.waitKey(300)
        cv2.destroyWindow("st")

        return construct_frame_image(), construct_fg_mask_image(), construct_fg_mask_count_image()

    with stitching_frames_fullpath(video).open("rb") as f:
        stitching_frames: list[StitchingFrame] = pickle.load(f)

    transforms = []
    for i_1 in range(len(stitching_frames) - 1):
        i_2 = i_1 + 1
        f_1 = stitching_frames[i_1]
        f_2 = stitching_frames[i_2]

        dx, dy = template_match(f_1, f_2, config)
        mat = np.eye(3)
        mat[0, 2] = -dx
        mat[1, 2] = -dy
        transforms.append(mat)

    #     cv2.imshow(
    #         "im",
    #         cv2.normalize(
    #             im_match,
    #             None,
    #             0,
    #             255,
    #             cv2.NORM_MINMAX,
    #             cv2.CV_8U,
    #         ),
    #     )
    #     print(dx, dy)
    #     cv2.waitKey(1)
    # cv2.destroyWindow("im")

    im_stitch, im_stitch_mask, im_count = _stitch_frames(
        w=stitching_frames[0].width,
        h=stitching_frames[0].height,
        frames=np.array([f.im for f in stitching_frames]),
        fg_masks=np.array([f.fg_mask for f in stitching_frames]),
        trans_lst=transforms,
    )

    return im_stitch, im_stitch_mask
