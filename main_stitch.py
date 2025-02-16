import cv2
import numpy as np
import pandas as pd

import repo.camera_param
import repo.global_config
import repo.laser_detection
import repo.laser_param
from model.video import Video
from util import video_service

VIDEO_NAMES: list[str] = "1,2,3,4,5".split(",")

results = []
for VIDEO_NAME in VIDEO_NAMES:
    video = Video(name=VIDEO_NAME)

    print("=" * 40)
    print("=" * 40)
    print(" " + str(video))
    print("=" * 40)
    print("=" * 40)

    video_service.clear_dynamic_data(video, fg=False, stitching_frames=False)
    config = video_service.StitchConfig(
        fg_w_edge=8,
        fg_feature_q_large=99,
        fg_feature_q_small=1,
        fg_n_end=10,
        fg_n_elim=3,
        fg_feature_eps=0.05,
        fg_feature_min_pts=15,
        fg_cov_lambda=0.01,
        fg_th_mahalanobis=30,
        fg_mask_pp_morph_open_px=2,
        fg_mask_pp_morph_close_px=10,
        tm_max_move=(800, 200),
        tm_n_horizontal_split=4,
        stitch_th_fg_overlap=1,
    )
    roi = repo.global_config.get().roi
    video_service.create_frames_roi(video, roi)
    video_service.create_foreground(video, config)

    camera_param = repo.camera_param.get(
        repo.global_config.get().active_profile_names.camera_param_profile_name,
    ).param
    laser_param = repo.laser_param.get(
        repo.global_config.get().active_profile_names.laser_param_profile_name,
    ).param
    laser_detection_model = repo.laser_detection.get(
        repo.global_config.get().active_profile_names.laser_detection_profile_name,
    ).model

    stitching_frames = video_service.get_stitching_frames(
        video,
        roi,
        camera_param,
        laser_param,
        laser_detection_model,
        n_mid_frames=1,
    )

    im_stitch, im_stitch_mask = video_service.get_stitched_image(video, config)
    contours, _ = cv2.findContours(
        im_stitch_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    x, y, w, h = cv2.boundingRect(contours[0])
    print(f"{w=}, {h=} [px]")

    z = np.array([np.median(f.points_world[:, 2]) for f in stitching_frames])  # (n,)
    z_mean = z.mean()
    cx, cy = camera_param.conversion_factor(
        u=(roi.screen_x_min + roi.screen_x_max) / 2,
        v=(roi.screen_y_min + roi.screen_y_max) / 2,
        z=z_mean,
    )
    print(f"{z=}, {z_mean=}, {cx=},{cy=}")

    w_mm = w * cx
    print(f"{w_mm=}")

    im = im_stitch.copy()
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("im", cv2.resize(im, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyWindow("im")

    results.append(w_mm)
    pd.DataFrame(results, index=VIDEO_NAMES[:len(results)]).to_csv("out.csv", encoding="cp932")
