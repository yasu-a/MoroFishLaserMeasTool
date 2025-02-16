import numpy as np

from model.camera_param import CameraParam
from model.global_config import ROI
from model.laser_param import LaserParam


def get_laser_2d_and_3d_points(
        laser_mask: np.ndarray,
        camera_param: CameraParam,
        laser_param: LaserParam,
        roi: ROI,
        screen_offset: tuple[int, int] = None,  # u, v  画像がもともとROIで切り取られているときに使う
) -> tuple[np.ndarray, np.ndarray]:  # [[u, v], ...], [[x, y, z], ...]
    assert laser_mask.dtype == np.uint8 and laser_mask.ndim == 2, \
        (laser_mask.dtype, laser_mask.shape)
    h, w = laser_mask.shape
    us = np.arange(w)
    vs = np.arange(h)

    # レーザー重心
    if screen_offset is None:
        x_roi_flag = np.where(
            roi.screen_x_predicate(us),
            1,
            0,
        )
    else:
        x_roi_flag = np.array([1] * len(us))
    laser_u_on_vertical = (laser_mask * us[None, :] * x_roi_flag[None, :]).sum(
        axis=1) / laser_mask.sum(axis=1)  # nan if undetected

    us = laser_u_on_vertical[~np.isnan(laser_u_on_vertical)]
    vs = vs[~np.isnan(laser_u_on_vertical)]

    if screen_offset is not None:
        u_ofs, v_ofs = screen_offset
        us += u_ofs
        vs += v_ofs

    # 3D空間上の点を求める
    xs, ys, zs = [], [], []
    for u, v in zip(us, vs):  # FIXME: optimize
        x, y, z = camera_param.from_2d_to_3d(
            u=u,
            v=v,
            a=float(laser_param.vec[0]),
            b=float(laser_param.vec[1]),
            c=float(laser_param.vec[2]),
        )
        xs.append(x)
        ys.append(y)
        zs.append(z)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # ROIで絞る
    cond = roi.screen_x_predicate(us)
    cond &= roi.screen_y_predicate(vs)
    cond &= roi.world_x_predicate(xs)
    cond &= roi.world_y_predicate(ys)
    cond &= roi.world_z_predicate(zs)
    xs = xs[cond]
    ys = ys[cond]
    zs = zs[cond]
    us = us[cond]
    vs = vs[cond]

    return np.array((us, vs), np.int16).T, np.array((xs, ys, zs), np.float32).T
